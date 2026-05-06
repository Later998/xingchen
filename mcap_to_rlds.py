#!/usr/bin/env python3
"""Convert Astribot MCAP trajectory to RLDS TFRecord dataset."""

import os
import struct
from typing import List, Optional, Tuple
from dataclasses import dataclass

import av
import cv2
import numpy as np
import tensorflow as tf
from mcap.reader import make_reader
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo


# ── Fixed joint ordering (28 joints total) ──────────────────────────
# Order: arm_left(7), arm_right(7), torso(2), head(2),
#        chassis(4), gripper_left(3), gripper_right(3)
JOINT_ORDER = [
    # arm_left (7)
    "arm_left_joint1", "arm_left_joint2", "arm_left_joint3",
    "arm_left_joint4", "arm_left_joint5", "arm_left_joint6",
    "arm_left_joint7",
    # arm_right (7)
    "arm_right_joint1", "arm_right_joint2", "arm_right_joint3",
    "arm_right_joint4", "arm_right_joint5", "arm_right_joint6",
    "arm_right_joint7",
    # torso (2)
    "torso_joint1", "torso_joint2",
    # head (2)
    "head_joint1", "head_joint2",
    # chassis (4)
    "chassis_joint1", "chassis_joint2", "chassis_joint3", "chassis_joint4",
    # gripper_left (3)
    "gripper_left_joint1", "gripper_left_joint2", "gripper_left_joint3",
    # gripper_right (3)
    "gripper_right_joint1", "gripper_right_joint2", "gripper_right_joint3",
]

JOINT_CHANNEL_MAP = {
    "arm_left":  "/astribot_arm_left/joint_space_states",
    "arm_right": "/astribot_arm_right/joint_space_states",
    "torso":     "/astribot_torso/joint_space_states",
    "head":      "/astribot_head/joint_space_states",
    "chassis":   "/astribot_chassis/joint_space_states",
    "gripper_left":  "/astribot_gripper_left/joint_space_states",
    "gripper_right": "/astribot_gripper_right/joint_space_states",
}

CMD_CHANNEL_MAP = {
    "arm_left":  "/astribot_arm_left/joint_space_command",
    "arm_right": "/astribot_arm_right/joint_space_command",
    "torso":     "/astribot_torso/joint_space_command",
    "head":      "/astribot_head/joint_space_command",
    "chassis":   "/astribot_chassis/joint_space_command",
    "gripper_left":  "/astribot_gripper_left/joint_space_command",
    "gripper_right": "/astribot_gripper_right/joint_space_command",
}

CAMERA_TOPICS = {
    "head_rgb":       "/astribot_camera/head_rgbd/color_compress/compressed/h264",
    "wrist_left":     "/astribot_camera/left_wrist_rgbd/color_compress/compressed/h264",
    "wrist_right":    "/astribot_camera/right_wrist_rgbd/color_compress/compressed/h264",
}

# Joint count per body group (must sum to 28)
GROUP_SIZES = {
    "arm_left": 7, "arm_right": 7,
    "torso": 2, "head": 2, "chassis": 4,
    "gripper_left": 3, "gripper_right": 3,
}

TF_TOPIC = "/tf"
POSE_CMD_TOPIC = "/astribot/pose_command_array"


@dataclass
class TimestampedMessage:
    stamp_ns: int           # timestamp in nanoseconds
    topic: str
    raw_data: bytes
    channel_id: int


def parse_ros_time(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Parse ROS time (secs, nsecs) from a little-endian byte buffer."""
    secs = struct.unpack_from("<I", data, offset)[0]
    nsecs = struct.unpack_from("<I", data, offset + 4)[0]
    return secs, nsecs


def ros_time_to_ns(data: bytes, offset: int = 0) -> int:
    secs, nsecs = parse_ros_time(data, offset)
    return secs * 1_000_000_000 + nsecs


class H264Decoder:
    """Stateful H264 decoder using PyAV. Maintains SPS/PPS context across frames."""

    def __init__(self):
        self.codec = av.CodecContext.create("h264", "r")

    def decode(self, data: bytes) -> Optional[np.ndarray]:
        """Decode one H264 access unit to BGR uint8 frame. Returns None on failure."""
        try:
            pkt = av.packet.Packet(data)
            frames = self.codec.decode(pkt)
            for frame in frames:
                if frame is not None:
                    return frame.to_ndarray(format="bgr24")
        except av.EOFError:
            pass
        except Exception:
            pass
        return None


def extract_string(data: bytes, offset: int) -> Tuple[str, int]:
    """Extract a ROS string (4-byte length + UTF-8 data) at offset."""
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    s = data[offset:offset + length].decode("utf-8")
    offset += length
    return s, offset


def get_joint_position(data: bytes) -> np.ndarray:
    """Parse astribot_msgs/RobotJointState → position float64 array."""
    offset = 12  # seq(4) + stamp(8)
    _, offset = extract_string(data, offset)  # frame_id
    # Skip mode (int8)
    offset += 1
    # Skip name string array
    name_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    for _ in range(name_count):
        _, offset = extract_string(data, offset)
    # Read position float64 array
    pos_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    return np.frombuffer(data, dtype=np.float64, count=pos_count, offset=offset)


def get_joint_command(data: bytes) -> np.ndarray:
    """Parse astribot_msgs/RobotJointController → command float64 array."""
    offset = 12  # seq(4) + stamp(8)
    _, offset = extract_string(data, offset)  # frame_id
    # Skip mode (int8)
    offset += 1
    # Skip name string array
    name_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    for _ in range(name_count):
        _, offset = extract_string(data, offset)
    # Read command float64 array
    cmd_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    return np.frombuffer(data, dtype=np.float64, count=cmd_count, offset=offset)


def get_pose_command_array(data: bytes) -> np.ndarray:
    """Parse /astribot/pose_command_array → astribot_msgs/RobotCartesianStates.
    Returns Nx19 array (x,y,z,qx,qy,qz,qw, vx,vy,vz, fx,fy,fz) or empty.
    """
    offset = 12  # seq(4) + stamp(8)
    _, offset = extract_string(data, offset)  # header.frame_id
    # names[] (string array)
    name_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    for _ in range(name_count):
        _, offset = extract_string(data, offset)
    # states[] — each RobotCartesianState: header + pose(7) + twist(6) + wrench(6)
    state_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    states = []
    for _ in range(state_count):
        off2 = offset + 12  # skip header seq(4) + stamp(8)
        _, off2 = extract_string(data, off2)  # frame_id
        pose = np.frombuffer(data, dtype=np.float64, count=7, offset=off2)
        off2 += 7 * 8
        twist = np.frombuffer(data, dtype=np.float64, count=6, offset=off2)
        off2 += 6 * 8
        wrench = np.frombuffer(data, dtype=np.float64, count=6, offset=off2)
        states.append(np.concatenate([pose, twist, wrench]))
        offset = off2 + 6 * 8
    return np.array(states) if states else np.empty((0, 19), dtype=np.float64)


def extract_static_tf(data: bytes) -> Optional[np.ndarray]:
    """Extract first transform from tf2_msgs/TFMessage as a 4x4 matrix."""
    offset = 0
    count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if count == 0:
        return None
    off = offset + 12  # skip seq(4) + stamp(8)
    _, off = extract_string(data, off)   # header.frame_id
    child_frame_id, off = extract_string(data, off)  # child_frame_id (unused)
    tx, ty, tz = struct.unpack_from("<ddd", data, off)
    off += 3 * 8
    qx, qy, qz, qw = struct.unpack_from("<dddd", data, off)
    mat = np.eye(4, dtype=np.float64)
    mat[0:3, 3] = [tx, ty, tz]
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    mat[0, 0] = 1 - 2 * (yy + zz)
    mat[0, 1] = 2 * (xy - wz)
    mat[0, 2] = 2 * (xz + wy)
    mat[1, 0] = 2 * (xy + wz)
    mat[1, 1] = 1 - 2 * (xx + zz)
    mat[1, 2] = 2 * (yz - wx)
    mat[2, 0] = 2 * (xz - wy)
    mat[2, 1] = 2 * (yz + wx)
    mat[2, 2] = 1 - 2 * (xx + yy)
    return mat


def read_all_messages(mcap_path: str):
    """Read all messages from MCAP, returning raw tagged messages.

    Camera messages store decoded protobuf (h264 bytes extracted).
    ROS1 messages store raw ROS1 serialized bytes.
    """
    camera_msgs: dict = {k: [] for k in CAMERA_TOPICS}
    state_msgs: List[TimestampedMessage] = []
    cmd_msgs: List[TimestampedMessage] = []
    pose_cmd_msgs: List[TimestampedMessage] = []
    tf_msgs: List[TimestampedMessage] = []
    state_topics = set(JOINT_CHANNEL_MAP.values())
    cmd_topics = set(CMD_CHANNEL_MAP.values())
    foxglove_schemas = {"foxglove.CompressedVideo"}

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            data = message.data
            encoding = schema.encoding if schema else ""

            # Foxglove protobuf messages
            if schema and schema.name in foxglove_schemas:
                video = CompressedVideo()
                video.ParseFromString(data)
                stamp_ns = 0
                if video.HasField("timestamp"):
                    stamp_ns = video.timestamp.seconds * 1_000_000_000 + video.timestamp.nanos
                if stamp_ns == 0:
                    stamp_ns = message.log_time
                tm = TimestampedMessage(
                    stamp_ns=stamp_ns,
                    topic=topic,
                    raw_data=video.data,  # store pure H264 bytes
                    channel_id=channel.id,
                )
                for cam_key, cam_topic in CAMERA_TOPICS.items():
                    if topic == cam_topic:
                        camera_msgs[cam_key].append(tm)
                        break
                continue

            # ROS1 messages
            stamp_ns = 0
            if len(data) >= 12:
                stamp_ns = ros_time_to_ns(data, 4)
            if stamp_ns == 0:
                stamp_ns = message.log_time

            tm = TimestampedMessage(
                stamp_ns=stamp_ns,
                topic=topic,
                raw_data=data,
                channel_id=channel.id,
            )

            if topic == TF_TOPIC:
                tf_msgs.append(tm)
            elif topic == POSE_CMD_TOPIC:
                pose_cmd_msgs.append(tm)
            elif topic in state_topics:
                state_msgs.append(tm)
            elif topic in cmd_topics:
                cmd_msgs.append(tm)
            # camera topics handled above; unknown topics skipped

    return camera_msgs, state_msgs, cmd_msgs, pose_cmd_msgs, tf_msgs


def temporal_nearest(stamp_ns: int, msgs: List[TimestampedMessage]) -> Optional[TimestampedMessage]:
    """Find the message with timestamp closest to stamp_ns (sorted input)."""
    if not msgs:
        return None
    lo, hi = 0, len(msgs)
    while lo < hi:
        mid = (lo + hi) // 2
        if msgs[mid].stamp_ns < stamp_ns:
            lo = mid + 1
        else:
            hi = mid
    idx = lo
    closest = None
    best_delta = None
    for candidate in [idx - 1, idx, idx + 1]:
        if 0 <= candidate < len(msgs):
            delta = abs(msgs[candidate].stamp_ns - stamp_ns)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                closest = msgs[candidate]
    return closest


def build_joint_vector(stamp_ns: int, msgs: List[TimestampedMessage],
                       channel_map: dict, parser) -> np.ndarray:
    """Build 28-D joint vector by nearest-neighbor alignment per joint group."""
    vec = np.zeros(28, dtype=np.float64)
    idx = 0
    for group, topic in channel_map.items():
        group_msgs = [m for m in msgs if m.topic == topic]
        nearest = temporal_nearest(stamp_ns, group_msgs)
        if nearest is not None:
            vals = parser(nearest.raw_data)
            n = GROUP_SIZES[group]
            vec[idx:idx + min(len(vals), n)] = vals[:n]
        idx += GROUP_SIZES[group]
    return vec


def decode_camera_image(
    stamp_ns: int,
    camera_stream: List[TimestampedMessage],
    decoder: H264Decoder,
) -> Optional[np.ndarray]:
    """Find nearest H264 frame in stream and decode to BGR uint8."""
    nearest = temporal_nearest(stamp_ns, camera_stream)
    if nearest is None:
        return None
    return decoder.decode(nearest.raw_data)


def sort_msgs(msgs):
    msgs.sort(key=lambda m: m.stamp_ns)


def main(mcap_path: str, output_path: str):
    episode_id = os.path.splitext(os.path.basename(mcap_path))[0]

    print(f"Reading MCAP: {mcap_path}")
    camera_msgs, state_msgs, cmd_msgs, pose_cmd_msgs, tf_msgs = read_all_messages(mcap_path)

    for k in camera_msgs:
        sort_msgs(camera_msgs[k])
    sort_msgs(state_msgs)
    sort_msgs(cmd_msgs)
    sort_msgs(pose_cmd_msgs)
    sort_msgs(tf_msgs)

    print(f"  Camera frames: {len(camera_msgs['head_rgb'])}")
    print(f"  State msgs:    {len(state_msgs)}")
    print(f"  Command msgs:  {len(cmd_msgs)}")
    print(f"  Pose cmd msgs: {len(pose_cmd_msgs)}")
    print(f"  TF msgs:       {len(tf_msgs)}")

    # Extract static transforms from first /tf message
    camera_extrinsics = np.eye(4, dtype=np.float64)
    for tm in tf_msgs:
        mat = extract_static_tf(tm.raw_data)
        if mat is not None:
            camera_extrinsics = mat
            break

    master_stream = camera_msgs["head_rgb"]
    total = len(master_stream)
    print(f"Master clock frames: {total}")
    print(f"Writing: {output_path}")

    # Per-camera H264 decoders (stateful, for SPS/PPS reference)
    decoders = {
        k: H264Decoder()
        for k in camera_msgs
    }

    writer = tf.io.TFRecordWriter(output_path)

    for i, master_msg in enumerate(master_stream):
        stamp_ns = master_msg.stamp_ns
        is_first = (i == 0)
        is_last = (i == total - 1)

        # ── Camera images (temporally aligned to master clock) ──
        img_head = decode_camera_image(stamp_ns, camera_msgs["head_rgb"], decoders["head_rgb"])
        img_wrist_l = decode_camera_image(stamp_ns, camera_msgs["wrist_left"], decoders["wrist_left"])
        img_wrist_r = decode_camera_image(stamp_ns, camera_msgs["wrist_right"], decoders["wrist_right"])

        # ── Joint state / command via temporal alignment ──
        joint_pos = build_joint_vector(stamp_ns, state_msgs, JOINT_CHANNEL_MAP, get_joint_position)
        joint_cmd = build_joint_vector(stamp_ns, cmd_msgs, CMD_CHANNEL_MAP, get_joint_command)

        # ── End-effector pose command ──
        nearest_pose = temporal_nearest(stamp_ns, pose_cmd_msgs)
        ee_pose = (
            get_pose_command_array(nearest_pose.raw_data)
            if nearest_pose is not None
            else np.empty((0, 19), dtype=np.float64)
        )

        # ── Serialize ──
        img_head_bytes = img_head.tobytes() if img_head is not None else b""
        img_wrist_l_bytes = img_wrist_l.tobytes() if img_wrist_l is not None else b""
        img_wrist_r_bytes = img_wrist_r.tobytes() if img_wrist_r is not None else b""

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "episode_metadata/episode_id":
                        bytes_feature(episode_id.encode()),
                    "episode_metadata/camera_extrinsics":
                        bytes_feature(camera_extrinsics.tobytes()),
                    "observation/image_head_rgb":
                        bytes_feature(img_head_bytes),
                    "observation/image_wrist_left_rgb":
                        bytes_feature(img_wrist_l_bytes),
                    "observation/image_wrist_right_rgb":
                        bytes_feature(img_wrist_r_bytes),
                    "observation/joint_positions":
                        float_list_feature(joint_pos.tolist()),
                    "action/joint_space_commands":
                        float_list_feature(joint_cmd.tolist()),
                    "action/end_effector_pose_commands":
                        float_list_feature(ee_pose.flatten().tolist()),
                    "language_instruction":
                        bytes_feature("Move to bedside".encode()),
                    "is_first": int64_feature(1 if is_first else 0),
                    "is_last": int64_feature(1 if is_last else 0),
                    "is_terminal": int64_feature(1 if is_last else 0),
                }
            )
        )
        writer.write(example.SerializeToString())

        if (i + 1) % 100 == 0 or is_last:
            print(f"  Progress: {i + 1}/{total}")

    writer.close()
    print(f"Done. Wrote {total} steps to {output_path}")


# ── TFRecord feature helpers ────────────────────────────────────────
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: mcap_to_rlds.py <input.mcap> <output.tfrecord>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
