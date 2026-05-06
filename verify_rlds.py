#!/usr/bin/env python3
"""Verify RLDS TFRecord structure - prints schema, shapes, types only."""

import numpy as np
import tensorflow as tf

tfrecord_path = "output.tfrecord"

dataset = tf.data.TFRecordDataset([tfrecord_path])
for raw in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    feats = example.features.feature

    print("=== Top-level Keys ===")
    for k in sorted(feats.keys()):
        print(f"  {k}")

    print("\n=== Detailed Inspection ===")

    # episode_metadata
    ep_id = feats["episode_metadata/episode_id"].bytes_list.value[0]
    extrinsics = feats["episode_metadata/camera_extrinsics"].bytes_list.value[0]
    extrinsics_arr = np.frombuffer(extrinsics, dtype=np.float64).reshape(4, 4)
    print(f"\n[episode_metadata]")
    print(f"  episode_id: {ep_id.decode()} (str)")
    print(f"  camera_extrinsics: shape {extrinsics_arr.shape}, dtype=float64")
    print(f"    [[{extrinsics_arr[0,0]:.3f}, {extrinsics_arr[0,1]:.3f}, ...]]")

    # observation
    print(f"\n[observation]")
    for img_key in ["image_head_rgb", "image_wrist_left_rgb", "image_wrist_right_rgb"]:
        raw = feats[f"observation/{img_key}"].bytes_list.value[0]
        if raw:
            arr = np.frombuffer(raw, dtype=np.uint8)
            n_pixels = arr.size // 3
            h = int(np.sqrt(n_pixels * 9 / 16))  # guess from 16:9
            w = n_pixels // h
            print(f"  {img_key}: {arr.size} bytes, dtype=uint8, shape ~({h}, {w}, 3)")
        else:
            print(f"  {img_key}: EMPTY")

    jpos = feats["observation/joint_positions"].float_list.value
    print(f"  joint_positions: len={len(jpos)}, dtype=float64, shape=({len(jpos)},)")

    # action
    print(f"\n[action]")
    jcmd = feats["action/joint_space_commands"].float_list.value
    print(f"  joint_space_commands: len={len(jcmd)}, dtype=float64, shape=({len(jcmd)},)")
    ee_pose = feats["action/end_effector_pose_commands"].float_list.value
    n_poses = len(ee_pose) // 19 if len(ee_pose) > 0 else 0
    print(f"  end_effector_pose_commands: len={len(ee_pose)}, dtype=float64, shape=({n_poses}, 19)")

    # language_instruction
    print(f"\n[language_instruction]")
    instr = feats["language_instruction"].bytes_list.value[0]
    print(f"  content: \"{instr.decode()}\"")

    # status flags
    print(f"\n[status flags]")
    for key in ["is_first", "is_last", "is_terminal"]:
        val = feats[key].int64_list.value[0]
        print(f"  {key}: {bool(val)}")

    print("\n=== Verification Complete ===")
