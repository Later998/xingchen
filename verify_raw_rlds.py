#!/usr/bin/env python3
"""Verify RLDS TFRecord structure: shapes, dtypes, and content."""

import os
import sys
import numpy as np
import tensorflow as tf

def main():
    path = "CJ260330-10-R195.tfrecord"
    if not os.path.isfile(path):
        path = os.path.join(os.path.dirname(__file__) or ".", path)
    if not os.path.isfile(path):
        path = os.path.join(os.path.expanduser("~"), path)

    dataset = tf.data.TFRecordDataset([path])

    for i, raw in enumerate(dataset):
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        feat = example.features.feature

        print("=" * 70)
        print(f"STEP {i}")
        print("=" * 70)

        # 1. observation keys
        print("\n--- observation ---")
        obs_keys = [k for k in feat.keys() if k.startswith("observation/")]
        for k in sorted(obs_keys):
            val = feat[k]
            kind = val.WhichOneof("kind")
            if kind == "bytes_list":
                data = val.bytes_list.value[0]
                arr = np.frombuffer(data, dtype=np.uint8)
                print(f"  {k}")
                if len(data) == 0:
                    print(f"    dtype: EMPTY")
                    continue
                # Try to infer shape from known keys
                if "head_rgb" in k and "depth" not in k:
                    arr = arr.view(np.uint8).reshape(720, 1280, 3)
                elif "head_depth" in k:
                    arr = np.frombuffer(data, dtype=np.float32).reshape(720, 1280, 1)
                elif "wrist_left_rgb" in k:
                    arr = arr.view(np.uint8).reshape(360, 640, 3)
                elif "wrist_left_depth" in k:
                    arr = np.frombuffer(data, dtype=np.float32).reshape(360, 640, 1)
                elif "wrist_right_rgb" in k:
                    arr = arr.view(np.uint8).reshape(360, 640, 3)
                elif "wrist_right_depth" in k:
                    arr = np.frombuffer(data, dtype=np.float32).reshape(360, 640, 1)
                else:
                    # generic: just report bytes
                    print(f"    bytes: {len(data)}")
                    print(f"    inferred_dtype: unknown")
                    continue
                print(f"    shape: {arr.shape}")
                print(f"    dtype: {arr.dtype}")
                print(f"    range: [{arr.min()}, {arr.max()}]")
                # show first few pixel values for depth
                if "depth" in k:
                    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
                    print(f"    sample[{cy},{cx}]: {arr[cy, cx, 0]:.1f}")
                    print(f"    nonzero_ratio: {(arr > 0).mean() * 100:.1f}%")
            elif kind == "float_list":
                vals = np.array(val.float_list.value)
                print(f"  {k}")
                print(f"    shape: ({len(vals)},)")
                print(f"    dtype: float64" if vals.dtype == np.float64 else f"    dtype: {vals.dtype}")
                print(f"    range: [{vals.min():.6f}, {vals.max():.6f}]")

        # 2. action keys
        print("\n--- action ---")
        act_keys = [k for k in feat.keys() if k.startswith("action/")]
        for k in sorted(act_keys):
            val = feat[k]
            kind = val.WhichOneof("kind")
            if kind == "bytes_list":
                data = val.bytes_list.value[0]
                print(f"  {k}: {len(data)} bytes")
            elif kind == "float_list":
                vals = np.array(val.float_list.value)
                print(f"  {k}")
                print(f"    shape: ({len(vals)},)")
                print(f"    dtype: {vals.dtype}")
                print(f"    range: [{vals.min():.6f}, {vals.max():.6f}]")

        # 3. language_instruction
        print("\n--- metadata ---")
        meta_keys = [k for k in feat.keys() if not k.startswith("observation/") and not k.startswith("action/")]
        for k in sorted(meta_keys):
            val = feat[k]
            kind = val.WhichOneof("kind")
            if kind == "bytes_list":
                data = val.bytes_list.value[0]
                if len(data) < 1024:
                    print(f"  {k}: \"{data.decode('utf-8', errors='replace')}\"")
                else:
                    print(f"  {k}: {len(data)} bytes")
            elif kind == "float_list":
                vals = np.array(val.float_list.value)
                print(f"  {k}: {vals.tolist()}")
            elif kind == "int64_list":
                vals = list(val.int64_list.value)
                print(f"  {k}: {vals}")

        print()
        break  # only first step

    # Also verify total step count
    total = sum(1 for _ in dataset)
    print(f"Total steps: {total}")

if __name__ == "__main__":
    main()
