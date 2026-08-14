#!/usr/bin/env python3
"""Select EXP-001 images from the authoritative AD-GS Waymo split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_training_frames(scene_path, count=4):
    scene_path = Path(scene_path).expanduser().resolve()
    image_dir = scene_path / "image"
    metadata_path = scene_path / "cameras.npz"
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )

    with np.load(metadata_path, allow_pickle=False) as metadata:
        is_val_list = np.asarray(metadata["is_val_list"], dtype=np.bool_)
        time_stamps = np.asarray(metadata["time_stamps"])

    if len(image_paths) != len(is_val_list) or len(image_paths) != len(time_stamps):
        raise ValueError(
            "image, is_val_list, and time_stamps lengths must match: "
            f"{len(image_paths)}, {len(is_val_list)}, {len(time_stamps)}"
        )

    selected_indices = np.flatnonzero(~is_val_list)[:count]
    if len(selected_indices) != count:
        raise ValueError(f"requested {count} training frames, found {len(selected_indices)}")

    return {
        "scene": str(scene_path),
        "metadata": str(metadata_path),
        "selection": "sorted image files paired with cameras.npz is_val_list == false",
        "count": count,
        "images": [
            {
                "index": int(index),
                "time_stamp": float(time_stamps[index]),
                "path": str(image_paths[index]),
                "size_bytes": image_paths[index].stat().st_size,
                "sha256": file_sha256(image_paths[index]),
            }
            for index in selected_indices
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("count must be positive")
    result = select_training_frames(args.scene, args.count)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
