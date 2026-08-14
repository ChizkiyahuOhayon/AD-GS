#!/usr/bin/env python3
"""Validate the locked AD-GS Waymo scene006 preprocessing contract."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_FRAME_COUNT = 86
REQUIRED_CAMERA_KEYS = ("R", "T", "K", "time_stamps", "is_val_list")
REQUIRED_PLY_PROPERTIES = {"x", "y", "z", "t"}
EXPECTED_CAMERA_SHAPES = {
    "R": (86, 3, 3),
    "T": (86, 3),
    "K": (86, 9),
    "time_stamps": (86,),
    "is_val_list": (86,),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_vertex_header(path):
    vertex_count = None
    vertex_properties = set()
    in_vertex_element = False
    with open(path, "rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("points3d.ply does not start with a PLY header")
        for raw_line in stream:
            line = raw_line.decode("ascii").strip()
            if line.startswith("element "):
                fields = line.split()
                in_vertex_element = fields[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(fields[2])
            elif in_vertex_element and line.startswith("property "):
                vertex_properties.add(line.split()[-1])
            elif line == "end_header":
                data_offset = stream.tell()
                break
        else:
            raise ValueError("points3d.ply is missing end_header")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError("points3d.ply must contain a positive vertex count")
    missing = REQUIRED_PLY_PROPERTIES - vertex_properties
    if missing:
        raise ValueError(f"points3d.ply is missing vertex properties: {sorted(missing)}")
    if path.stat().st_size <= data_offset:
        raise ValueError("points3d.ply has no vertex payload")
    return vertex_count, sorted(vertex_properties)


def validate_scene(scene_path):
    scene_path = Path(scene_path).expanduser().resolve()
    image_dir = scene_path / "image"
    metadata_path = scene_path / "cameras.npz"
    points_path = scene_path / "points3d.ply"
    image_paths = sorted(image_dir.glob("*.jpg"))
    expected_names = [f"{index:06d}.jpg" for index in range(EXPECTED_FRAME_COUNT)]

    if [path.name for path in image_paths] != expected_names:
        raise ValueError("scene006 must contain exactly 000000.jpg through 000085.jpg")

    image_records = []
    for path in image_paths:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        image_records.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "width": width,
                "height": height,
                "mode": mode,
            }
        )

    with np.load(metadata_path, allow_pickle=False) as metadata:
        missing_keys = sorted(set(REQUIRED_CAMERA_KEYS) - set(metadata.files))
        if missing_keys:
            raise ValueError(f"cameras.npz is missing keys: {missing_keys}")
        arrays = {key: np.asarray(metadata[key]) for key in REQUIRED_CAMERA_KEYS}

    for key, value in arrays.items():
        if value.shape != EXPECTED_CAMERA_SHAPES[key]:
            raise ValueError(
                f"cameras.npz {key} shape must be {EXPECTED_CAMERA_SHAPES[key]}, "
                f"got {value.shape}"
            )
        if key != "is_val_list" and not np.isfinite(value).all():
            raise ValueError(f"cameras.npz {key} contains nonfinite values")
    if arrays["is_val_list"].dtype != np.bool_:
        raise ValueError("cameras.npz is_val_list must have boolean dtype")

    expected_timestamps = np.arange(EXPECTED_FRAME_COUNT, dtype=np.float32)
    if not np.array_equal(arrays["time_stamps"], expected_timestamps):
        raise ValueError("time_stamps must equal 0 through 85")

    expected_is_val = np.zeros(EXPECTED_FRAME_COUNT, dtype=np.bool_)
    expected_is_val[4::4] = True
    is_val = arrays["is_val_list"].astype(np.bool_)
    if not np.array_equal(is_val, expected_is_val):
        raise ValueError("is_val_list does not match the released every-fourth-frame split")

    vertex_count, vertex_properties = read_vertex_header(points_path)
    return {
        "scene": str(scene_path),
        "frame_count": EXPECTED_FRAME_COUNT,
        "validation_indices": np.flatnonzero(is_val).tolist(),
        "training_indices": np.flatnonzero(~is_val).tolist(),
        "cameras": {
            "path": str(metadata_path),
            "size_bytes": metadata_path.stat().st_size,
            "sha256": file_sha256(metadata_path),
            "shapes": {key: list(value.shape) for key, value in arrays.items()},
        },
        "points3d": {
            "path": str(points_path),
            "size_bytes": points_path.stat().st_size,
            "sha256": file_sha256(points_path),
            "vertex_count": vertex_count,
            "vertex_properties": vertex_properties,
        },
        "images": image_records,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = validate_scene(args.scene)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
