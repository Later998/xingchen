#!/usr/bin/env python3
"""Convert Astribot MCAP trajectory to RLDS TFRecord dataset.

Supports:
  - Foxglove CompressedVideo (H264) for RGB (backward compat)
  - sensor_msgs/CompressedImage (JPEG) for RGB
  - sensor_msgs/CompressedImage (raw uint16) for depth
  - Dynamic language_instruction from CLI/env/file
"""

import os
import sys
import struct
from typing import List, Optional, Tuple, Dict
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

# Device name → output key mapping for camera topics
# Topic pattern: /astribot_camera/{device}/...
DEVICE_OUTPUT_MAP = {
    "head_rgbd":      "head",
    "left_wrist_rgbd": "wrist_left",
    "right_wrist_rgbd": "wrist_right",
    "torso_rgbd":     "torso",
    "head_stereo":    "head_stereo",
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


@dataclass
class CameraMessage:
    stamp_ns: int
    topic: str
    raw_data: bytes         # payload varies by format
    cam_type: str           # "foxglove_h264", "compressed_jpeg", "raw_depth"
    format_str: str         # "" / "jpeg" / "raw"
    device_name: str        # e.g. "head_rgbd"


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


def extract_string(data: bytes, offset: int, cdr_align: bool = False) -> Tuple[str, int]:
    """Extract a ROS string (4-byte length + UTF-8 data) at offset.

    If cdr_align is True, reads with 4-byte pre-alignment and post-padding.
    """
    if cdr_align:
        offset = (offset + 3) & ~3
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    s = data[offset:offset + length].decode("utf-8")
    offset += length
    return s, offset


def _is_ros2_cdr(data: bytes) -> bool:
    """Detect ROS2 CDR encapsulation (0x00 0x01 prefix)."""
    return bool(len(data) >= 4 and data[0] == 0x00 and data[1] == 0x01)


def _skip_header_cdr(data: bytes, off: int) -> int:
    """Skip std_msgs/Header in ROS2 CDR: stamp(int32+uint32) + frame_id(string)."""
    off += 4 + 4  # sec + nanosec
    off = (off + 3) & ~3
    flen = struct.unpack_from("<I", data, off)[0]; off += 4
    off += flen
    return off


def get_joint_position(data: bytes) -> np.ndarray:
    """Parse astribot_msgs/RobotJointState → position float64 array.

    Handles ROS1 and ROS2 CDR serialization.
    """
    is_ros2 = _is_ros2_cdr(data)
    if is_ros2:
        off = 4  # skip CDR encapsulation
        off = _skip_header_cdr(data, off)  # header.stamp + header.frame_id
        mode = struct.unpack_from("<b", data, off)[0]; off += 1  # int8, 1-byte align
        off = (off + 3) & ~3  # align name[] count
    else:
        off = 12  # seq(4) + stamp(8)
        _, off = extract_string(data, off)  # frame_id
        off += 1  # mode

    name_count = struct.unpack_from("<I", data, off)[0]; off += 4
    for _ in range(name_count):
        _, off = extract_string(data, off, cdr_align=is_ros2)
    off = (off + 3) & ~3  # align position[] count
    pos_count = struct.unpack_from("<I", data, off)[0]; off += 4
    return np.frombuffer(data, dtype=np.float64, count=pos_count, offset=off)


def get_joint_command(data: bytes) -> np.ndarray:
    """Parse astribot_msgs/RobotJointController → command float64 array.

    Handles ROS1 and ROS2 CDR serialization.
    """
    is_ros2 = _is_ros2_cdr(data)
    if is_ros2:
        off = 4
        off = _skip_header_cdr(data, off)
        mode = struct.unpack_from("<b", data, off)[0]; off += 1
        off = (off + 3) & ~3
    else:
        off = 12
        _, off = extract_string(data, off)
        off += 1

    name_count = struct.unpack_from("<I", data, off)[0]; off += 4
    for _ in range(name_count):
        _, off = extract_string(data, off, cdr_align=is_ros2)
    off = (off + 3) & ~3
    cmd_count = struct.unpack_from("<I", data, off)[0]; off += 4
    return np.frombuffer(data, dtype=np.float64, count=cmd_count, offset=off)


def get_pose_command_array(data: bytes) -> np.ndarray:
    """Parse /astribot/pose_command_array → Nx19 array.

    Handles ROS1 and ROS2 CDR serialization.
    """
    is_ros2 = _is_ros2_cdr(data)
    if is_ros2:
        off = 4
        off = _skip_header_cdr(data, off)
        off = (off + 3) & ~3
    else:
        off = 12
        _, off = extract_string(data, off)

    name_count = struct.unpack_from("<I", data, off)[0]; off += 4
    for _ in range(name_count):
        _, off = extract_string(data, off, cdr_align=is_ros2)

    if is_ros2:
        off = (off + 3) & ~3
    state_count = struct.unpack_from("<I", data, off)[0]; off += 4
    states = []
    for _ in range(state_count):
        if is_ros2:
            # each state: header (stamp+frame_id) then pose(7)+twist(6)+wrench(6)
            off2 = _skip_header_cdr(data, off)
            off2 = (off2 + 3) & ~3  # align to float64 (ROS2 Fast CDR uses 4-byte align, not 8)
        else:
            off2 = off + 12
            _, off2 = extract_string(data, off2)
        pose = np.frombuffer(data, dtype=np.float64, count=7, offset=off2)
        off2 += 7 * 8
        twist = np.frombuffer(data, dtype=np.float64, count=6, offset=off2)
        off2 += 6 * 8
        wrench = np.frombuffer(data, dtype=np.float64, count=6, offset=off2)
        states.append(np.concatenate([pose, twist, wrench]))
        off = off2 + 6 * 8
    return np.array(states) if states else np.empty((0, 19), dtype=np.float64)


def extract_static_tf(data: bytes) -> Optional[np.ndarray]:
    """Extract first transform from tf2_msgs/TFMessage as a 4x4 matrix.

    Handles both ROS1 and ROS2 CDR serialization.
    """
    off = 0
    is_ros2 = bool(len(data) >= 4 and data[0] == 0x00 and data[1] == 0x01)
    if is_ros2:
        off = 4  # skip CDR encapsulation header

    count = struct.unpack_from("<I", data, off)[0]
    off += 4
    if count == 0:
        return None

    if is_ros2:
        off = (off + 3) & ~3
        _ = struct.unpack_from("<i", data, off)[0]; off += 4  # sec
        _ = struct.unpack_from("<I", data, off)[0]; off += 4  # nsec
        off = (off + 3) & ~3
        flen = struct.unpack_from("<I", data, off)[0]; off += 4
        off += flen
        off = (off + 3) & ~3
        cflen = struct.unpack_from("<I", data, off)[0]; off += 4
        off += cflen
        off = (off + 3) & ~3  # align to float64 (ROS2 Fast CDR 4-byte alignment)
    else:
        off += 12
        _, off = extract_string(data, off)
        _, off = extract_string(data, off)

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


# ── sensor_msgs/CompressedImage ROS2 CDR parser ────────────────────

def parse_compressed_image(data: bytes) -> Tuple[int, int, str, str, bytes]:
    """Parse ROS2 CDR serialized sensor_msgs/CompressedImage.

    Returns: (sec, nanosec, frame_id, format_str, image_bytes)
    """
    off = 4  # skip CDR encapsulation header (0x00 0x01 0x00 0x00)
    sec = struct.unpack_from("<i", data, off)[0]; off += 4
    nsec = struct.unpack_from("<I", data, off)[0]; off += 4
    off = (off + 3) & ~3
    flen = struct.unpack_from("<I", data, off)[0]; off += 4
    frame_id = data[off:off + flen].decode("utf-8", errors="replace")
    off += flen
    off = (off + 3) & ~3
    fmt_len = struct.unpack_from("<I", data, off)[0]; off += 4
    fmt = data[off:off + fmt_len].decode("utf-8", errors="replace").rstrip("\x00")
    off += fmt_len
    off = (off + 3) & ~3
    img_len = struct.unpack_from("<I", data, off)[0]; off += 4
    img_bytes = data[off:off + img_len]
    return sec, nsec, frame_id, fmt, img_bytes


# ── Depth dimension auto-detection ──────────────────────────────────

_DEPTH_DIMS_CACHE: Dict[str, Tuple[int, int]] = {}


def detect_raw_depth_dims(pixels: np.ndarray, hint_w: int = 0) -> Tuple[int, int]:
    """Auto-detect (height, width) of a raw uint16 depth image.

    Uses row-continuity correlation to find the most likely stride.
    If hint_w is provided (e.g. from RGB width), prefer widths close to it.
    Result is cached globally to avoid re-detection per frame.
    """
    npx = len(pixels)
    if npx == 0:
        return (0, 0)

    best_w = None
    best_score = -999.0
    for w in range(320, 2049, 8):
        if npx % w != 0 or w > npx:
            continue
        x = pixels[:-1][(np.arange(npx - 1) % w) != (w - 1)]
        y = pixels[1:][(np.arange(0, npx - 1) % w) != (w - 1)]
        if len(x) < 100:
            continue
        xm = x.astype(np.float64) - x.mean()
        ym = y.astype(np.float64) - y.mean()
        denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
        if denom < 1e-12:
            continue
        corr = (xm * ym).sum() / denom
        # Score: correlation + bonus for matching hint_w aspect
        score = corr
        if hint_w > 0:
            # Prefer widths close to hint_w (e.g. RGB width or half/quarter of it)
            penalty = abs(w - hint_w) / max(hint_w, 1) * 0.05
            score -= penalty
        # Penalize extreme aspect ratios
        h = npx // w
        aspect = max(w, h) / min(w, h)
        if aspect > 2.0:
            score -= 0.1 * (aspect - 2.0)
        if score > best_score:
            best_score = score
            best_w = w

    if best_w is None:
        best_w = int(np.sqrt(npx))
        while npx % best_w != 0:
            best_w -= 1

    h = npx // best_w
    return (h, best_w)


def get_depth_dims(device: str, raw_bytes: bytes, hint_w: int = 0) -> Tuple[int, int]:
    """Get depth image dimensions for a device, auto-detecting on first call."""
    if device not in _DEPTH_DIMS_CACHE:
        pixels = np.frombuffer(raw_bytes, dtype=np.uint16)
        h, w = detect_raw_depth_dims(pixels, hint_w=hint_w)
        _DEPTH_DIMS_CACHE[device] = (h, w)
    return _DEPTH_DIMS_CACHE[device]


# ── Image / depth decode helpers ────────────────────────────────────

def decode_compressed_jpeg(data: bytes) -> Optional[np.ndarray]:
    """Decode JPEG bytes to BGR uint8 image (HxWx3)."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None else None


def decode_raw_depth(data: bytes, device: str) -> Optional[np.ndarray]:
    """Decode raw uint16 depth bytes to float32 image (HxWx1).

    Dimensions are auto-detected. Invalid (zero) depth remains zero.
    """
    if len(data) < 2:
        return None
    pixels = np.frombuffer(data, dtype=np.uint16).astype(np.float32)
    h, w = get_depth_dims(device, data)
    if h * w != len(pixels):
        return None
    return pixels.reshape(h, w, 1)


# ── Camera topic auto-detection ─────────────────────────────────────

def build_camera_maps(mcap_path: str):
    """Scan MCAP summary to detect RGB and depth topics.

    Returns:
      rgb_topics: dict[device_name] -> (cam_type, topic_full_path)
      depth_topics: dict[device_name] -> (cam_type, topic_full_path)
    """
    rgb_topics: Dict[str, Tuple[str, str]] = {}
    depth_topics: Dict[str, Tuple[str, str]] = {}

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if not summary:
            return rgb_topics, depth_topics

        for ch_id, channel in summary.channels.items():
            topic = channel.topic
            schema = summary.schemas.get(channel.schema_id)
            if not schema:
                continue
            name = schema.name
            device = ""
            if "/astribot_camera/" in topic:
                device = topic.split("/")[2]

            # Detect RGB sources
            if "/color_compress/compressed" in topic:
                if name == "foxglove.CompressedVideo" or "foxglove" in name:
                    rgb_topics[device] = ("foxglove_h264", topic)
                elif name == "sensor_msgs/msg/CompressedImage":
                    rgb_topics[device] = ("compressed_jpeg", topic)

            # Detect depth sources
            if "depth_compress" in topic:
                depth_topics[device] = ("raw_depth", topic)

    return rgb_topics, depth_topics


# ── MCAP reading ────────────────────────────────────────────────────

def read_all_messages(mcap_path: str, rgb_map: dict, depth_map: dict):
    """Read all messages from MCAP, returning typed containers.

    Returns:
      camera_msgs: dict[device] -> list of CameraMessage  (RGB)
      depth_msgs:  dict[device] -> list of CameraMessage  (depth)
      state_msgs, cmd_msgs, pose_cmd_msgs, tf_msgs: lists of TimestampedMessage
    """
    camera_msgs: dict = {}
    depth_msgs: dict = {}
    state_msgs: List[TimestampedMessage] = []
    cmd_msgs: List[TimestampedMessage] = []
    pose_cmd_msgs: List[TimestampedMessage] = []
    tf_msgs: List[TimestampedMessage] = []
    state_topics = set(JOINT_CHANNEL_MAP.values())
    cmd_topics = set(CMD_CHANNEL_MAP.values())

    # Build lookup: topic -> (cam_type, device) for known camera topics
    rgb_lookup = {}
    for dev, (ctype, topic) in rgb_map.items():
        rgb_lookup[topic] = (ctype, dev)
    depth_lookup = {}
    for dev, (ctype, topic) in depth_map.items():
        depth_lookup[topic] = (ctype, dev)

    # Build reverse: topic -> device for output key mapping
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            data = message.data
            schema_name = schema.name if schema else ""

            # ── Foxglove CompressedVideo (H264) ──
            if schema_name == "foxglove.CompressedVideo":
                video = CompressedVideo()
                video.ParseFromString(data)
                stamp_ns = 0
                if video.HasField("timestamp"):
                    stamp_ns = video.timestamp.seconds * 1_000_000_000 + video.timestamp.nanos
                if stamp_ns == 0:
                    stamp_ns = message.log_time
                tm = CameraMessage(
                    stamp_ns=stamp_ns,
                    topic=topic,
                    raw_data=video.data,
                    cam_type="foxglove_h264",
                    format_str="",
                    device_name=rgb_lookup.get(topic, (None, None))[1] or "",
                )
                if tm.device_name in rgb_map:
                    camera_msgs.setdefault(tm.device_name, []).append(tm)
                continue

            # ── sensor_msgs/CompressedImage (RGB JPEG or depth raw) ──
            if schema_name == "sensor_msgs/msg/CompressedImage":
                sec, nsec, frame_id, fmt, img_bytes = parse_compressed_image(data)
                stamp_ns = sec * 1_000_000_000 + nsec
                if stamp_ns == 0:
                    stamp_ns = message.log_time

                # Check if it's a known depth topic
                if topic in depth_lookup:
                    ctype, dev = depth_lookup[topic]
                    cm = CameraMessage(
                        stamp_ns=stamp_ns,
                        topic=topic,
                        raw_data=img_bytes,
                        cam_type="raw_depth",
                        format_str=fmt,
                        device_name=dev,
                    )
                    depth_msgs.setdefault(dev, []).append(cm)

                # Check if it's a known RGB topic
                if topic in rgb_lookup:
                    ctype, dev = rgb_lookup[topic]
                    cm = CameraMessage(
                        stamp_ns=stamp_ns,
                        topic=topic,
                        raw_data=img_bytes,
                        cam_type="compressed_jpeg",
                        format_str=fmt,
                        device_name=dev,
                    )
                    camera_msgs.setdefault(dev, []).append(cm)
                continue

            # ── ROS1 serialized messages ──
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

    return camera_msgs, depth_msgs, state_msgs, cmd_msgs, pose_cmd_msgs, tf_msgs


# ── Temporal alignment ──────────────────────────────────────────────

def temporal_nearest(stamp_ns: int, msgs: List) -> Optional:
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


def decode_rgb_frame(stamp_ns: int, camera_stream: List[CameraMessage],
                     decoder: H264Decoder, cam_type: str) -> Optional[np.ndarray]:
    """Find nearest frame and decode based on camera type."""
    nearest = temporal_nearest(stamp_ns, camera_stream)
    if nearest is None:
        return None
    if nearest.cam_type == "foxglove_h264":
        return decoder.decode(nearest.raw_data)
    elif nearest.cam_type == "compressed_jpeg":
        return decode_compressed_jpeg(nearest.raw_data)
    return None


def get_depth_at_stamp(stamp_ns: int, depth_stream: List[CameraMessage],
                       device: str) -> Optional[np.ndarray]:
    """Nearest-neighbor depth aligned to master clock, resized to (H, W, 1)."""
    nearest = temporal_nearest(stamp_ns, depth_stream)
    if nearest is None:
        return None
    return decode_raw_depth(nearest.raw_data, device)


def sort_camera_msgs(msgs):
    msgs.sort(key=lambda m: m.stamp_ns)


def sort_msgs(msgs):
    msgs.sort(key=lambda m: m.stamp_ns)


# ── Language instruction resolution ─────────────────────────────────

def resolve_language_instruction() -> str:
    """Resolve language instruction from CLI arg, env var, or file.

    Priority: --instruction CLI arg > LANGUAGE_INSTRUCTION env var
              > --language-file > LANGUAGE_FILE env var > 'Unknown task'
    """
    # Check CLI for --instruction
    if "--instruction" in sys.argv:
        idx = sys.argv.index("--instruction")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]

    # Check env var
    env_inst = os.environ.get("LANGUAGE_INSTRUCTION")
    if env_inst:
        return env_inst

    # Check --language-file
    if "--language-file" in sys.argv:
        idx = sys.argv.index("--language-file")
        if idx + 1 < len(sys.argv):
            fpath = sys.argv[idx + 1]
            try:
                with open(fpath, "r") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except (FileNotFoundError, IOError):
                pass

    # Check LANGUAGE_FILE env var
    env_file = os.environ.get("LANGUAGE_FILE")
    if env_file:
        try:
            with open(env_file, "r") as f:
                content = f.read().strip()
                if content:
                    return content
        except (FileNotFoundError, IOError):
            pass

    return "Unknown task"


# ── Main ────────────────────────────────────────────────────────────

def main(mcap_path: str, output_path: str):
    episode_id = os.path.splitext(os.path.basename(mcap_path))[0]
    language_instruction = resolve_language_instruction()
    print(f"Language instruction: '{language_instruction}'")

    # ── Auto-detect camera topics ──
    print(f"Reading MCAP: {mcap_path}")
    rgb_map, depth_map = build_camera_maps(mcap_path)
    print(f"  Detected RGB cameras: {list(rgb_map.keys())}")
    print(f"  Detected depth cameras: {list(depth_map.keys())}")

    # ── Read all messages ──
    camera_msgs, depth_msgs, state_msgs, cmd_msgs, pose_cmd_msgs, tf_msgs = \
        read_all_messages(mcap_path, rgb_map, depth_map)

    for dev in camera_msgs:
        sort_camera_msgs(camera_msgs[dev])
    for dev in depth_msgs:
        sort_camera_msgs(depth_msgs[dev])
    sort_msgs(state_msgs)
    sort_msgs(cmd_msgs)
    sort_msgs(pose_cmd_msgs)
    sort_msgs(tf_msgs)

    print(f"  Camera frames:")
    for dev in sorted(camera_msgs):
        print(f"    {dev} (RGB): {len(camera_msgs[dev])} frames")
    for dev in sorted(depth_msgs):
        print(f"    {dev} (depth): {len(depth_msgs[dev])} frames")
    print(f"  State msgs:    {len(state_msgs)}")
    print(f"  Command msgs:  {len(cmd_msgs)}")
    print(f"  Pose cmd msgs: {len(pose_cmd_msgs)}")
    print(f"  TF msgs:       {len(tf_msgs)}")

    # ── Static transforms ──
    camera_extrinsics = np.eye(4, dtype=np.float64)
    for tm in tf_msgs:
        mat = extract_static_tf(tm.raw_data)
        if mat is not None:
            camera_extrinsics = mat
            break

    # ── Determine master clock (head RGB timestamps) ──
    # Find head camera key
    head_key = None
    for dev in camera_msgs:
        out_key = DEVICE_OUTPUT_MAP.get(dev, dev)
        if out_key == "head":
            head_key = dev
            break
    if head_key is None and camera_msgs:
        head_key = next(iter(camera_msgs))
        print(f"  WARNING: no 'head' camera found, using '{head_key}' as master clock")

    if head_key is None:
        print("  ERROR: no RGB camera streams available!")
        return

    master_stream = camera_msgs[head_key]
    total = len(master_stream)
    print(f"Master clock frames (from '{head_key}' RGB): {total}")
    print(f"Writing: {output_path}")

    # ── Per-camera H264 decoders (only for foxglove_h264 cameras) ──
    decoders = {}
    for dev in camera_msgs:
        ctype, _ = rgb_map.get(dev, ("", ""))
        if ctype == "foxglove_h264":
            decoders[dev] = H264Decoder()

    # ── RGB shapes discovered lazily (for depth resize) ──
    rgb_shapes: Dict[str, Tuple[int, int, int]] = {}

    writer = tf.io.TFRecordWriter(output_path)

    # ── Pre-detect depth dimensions ──
    for dev, stream in depth_msgs.items():
        if not stream:
            continue
        hint_w = 0
        # Try to get RGB width hint from matching camera
        if dev in camera_msgs and camera_msgs[dev]:
            first = camera_msgs[dev][0]
            if first.cam_type == "compressed_jpeg":
                frame = decode_compressed_jpeg(first.raw_data)
                if frame is not None:
                    hint_w = frame.shape[1]
        get_depth_dims(dev, stream[0].raw_data, hint_w=hint_w)

    for i, master_msg in enumerate(master_stream):
        stamp_ns = master_msg.stamp_ns
        is_first = (i == 0)
        is_last = (i == total - 1)

        # ── RGB images (temporally aligned to master clock) ──
        images = {}
        for dev in camera_msgs:
            out_key = DEVICE_OUTPUT_MAP.get(dev, dev)
            ctype, _ = rgb_map.get(dev, ("", ""))
            decoder = decoders.get(dev)
            img = decode_rgb_frame(stamp_ns, camera_msgs[dev], decoder, ctype)
            if img is not None and dev not in rgb_shapes:
                rgb_shapes[dev] = img.shape
                rgb_shapes[out_key] = img.shape  # also store under output key
            images[out_key] = img

        # ── Depth images (temporally aligned to master clock) ──
        depths = {}
        for dev in depth_msgs:
            out_key = DEVICE_OUTPUT_MAP.get(dev, dev)
            depth = get_depth_at_stamp(stamp_ns, depth_msgs[dev], dev)
            if depth is not None and out_key in rgb_shapes:
                # Resize depth to match RGB resolution
                h_rgb, w_rgb = rgb_shapes[out_key][:2]
                if depth.shape[:2] != (h_rgb, w_rgb):
                    depth = cv2.resize(depth, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
            depths[out_key] = depth

        # ── Joint state / command via temporal alignment ──
        joint_pos = build_joint_vector(stamp_ns, state_msgs, JOINT_CHANNEL_MAP, get_joint_position)
        joint_cmd = build_joint_vector(stamp_ns, cmd_msgs, CMD_CHANNEL_MAP, get_joint_command)

        # ── End-effector pose command ──
        nearest_pose = temporal_nearest(stamp_ns, pose_cmd_msgs)
        ee_pose = np.empty((0, 19), dtype=np.float64)
        if nearest_pose is not None:
            try:
                ee_pose = get_pose_command_array(nearest_pose.raw_data)
            except Exception as e:
                print(f"  WARNING: pose cmd parse failed at step {i}: {e}")
                ee_pose = np.empty((0, 19), dtype=np.float64)

        # ── Serialize ──
        rgb_fields = {
            "head": images.get("head"),
            "wrist_left": images.get("wrist_left"),
            "wrist_right": images.get("wrist_right"),
        }
        depth_fields = {
            "head": depths.get("head"),
            "wrist_left": depths.get("wrist_left"),
            "wrist_right": depths.get("wrist_right"),
        }

        feature = {
            "episode_metadata/episode_id":
                bytes_feature(episode_id.encode()),
            "episode_metadata/camera_extrinsics":
                bytes_feature(camera_extrinsics.tobytes()),
            "observation/image_head_rgb":
                bytes_feature(rgb_fields["head"].tobytes() if rgb_fields["head"] is not None else b""),
            "observation/image_wrist_left_rgb":
                bytes_feature(rgb_fields["wrist_left"].tobytes() if rgb_fields["wrist_left"] is not None else b""),
            "observation/image_wrist_right_rgb":
                bytes_feature(rgb_fields["wrist_right"].tobytes() if rgb_fields["wrist_right"] is not None else b""),
            "observation/image_head_depth":
                bytes_feature(depth_fields["head"].tobytes() if depth_fields["head"] is not None else b""),
            "observation/image_wrist_left_depth":
                bytes_feature(depth_fields["wrist_left"].tobytes() if depth_fields["wrist_left"] is not None else b""),
            "observation/image_wrist_right_depth":
                bytes_feature(depth_fields["wrist_right"].tobytes() if depth_fields["wrist_right"] is not None else b""),
            "observation/joint_positions":
                float_list_feature(joint_pos.tolist()),
            "action/joint_space_commands":
                float_list_feature(joint_cmd.tolist()),
            "action/end_effector_pose_commands":
                float_list_feature(ee_pose.flatten().tolist()),
            "language_instruction":
                bytes_feature(language_instruction.encode()),
            "is_first": int64_feature(1 if is_first else 0),
            "is_last": int64_feature(1 if is_last else 0),
            "is_terminal": int64_feature(1 if is_last else 0),
        }

        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        writer.write(example.SerializeToString())

        if (i + 1) % 100 == 0 or is_last:
            print(f"  Progress: {i + 1}/{total}")

    writer.close()
    print(f"Done. Wrote {total} steps to {output_path}")

    # Print depth dimension info
    if _DEPTH_DIMS_CACHE:
        print(f"\nDetected depth dimensions:")
        for dev, (h, w) in sorted(_DEPTH_DIMS_CACHE.items()):
            out_key = DEVICE_OUTPUT_MAP.get(dev, dev)
            print(f"  {out_key} ({dev}): {h}x{w} uint16")


# ── TFRecord feature helpers ────────────────────────────────────────
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# ── CLI ─────────────────────────────────────────────────────────────
def print_usage():
    print("Usage: mcap_to_rlds.py <input.mcap> <output.tfrecord> [options]")
    print("Options:")
    print("  --instruction <text>         Language instruction (takes priority)")
    print("  --language-file <path>       Read language instruction from file")
    print("  LANGUAGE_INSTRUCTION env var  Fallback language instruction")
    print("  LANGUAGE_FILE env var         Fallback language instruction file")


if __name__ == "__main__":
    # Strip known optional args before positional check
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(positional) < 2:
        print_usage()
        sys.exit(1)

    mcap_path = positional[0]
    output_path = positional[1]

    if not os.path.isfile(mcap_path):
        print(f"Error: input file not found: {mcap_path}")
        sys.exit(1)

    main(mcap_path, output_path)
