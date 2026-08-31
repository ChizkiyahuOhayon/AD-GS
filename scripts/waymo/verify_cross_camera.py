import argparse
import json
from pathlib import Path

import numpy as np


def audit_cross_camera_scene(scene_path):
    scene_path = Path(scene_path)
    metadata = np.load(scene_path / "cameras.npz")
    required = {"camera_ids", "is_val_list", "time_stamps"}
    missing = sorted(required - set(metadata.files))
    if missing:
        raise ValueError("missing camera metadata: {}".format(missing))

    camera_ids = np.asarray(metadata["camera_ids"], dtype=np.int64)
    is_validation = np.asarray(metadata["is_val_list"], dtype=np.bool_)
    time_stamps = np.asarray(metadata["time_stamps"])
    if not (camera_ids.shape == is_validation.shape == time_stamps.shape):
        raise ValueError("camera_ids, is_val_list, and time_stamps must align")

    image_count = len(list((scene_path / "image").glob("*.jpg")))
    if image_count != camera_ids.size:
        raise ValueError("camera metadata and image counts differ")

    expected_cameras = {0, 1, 2}
    actual_cameras = set(camera_ids.tolist())
    if actual_cameras != expected_cameras:
        raise ValueError(
            "expected cameras [0, 1, 2], found {}".format(sorted(actual_cameras))
        )
    if set(camera_ids[~is_validation].tolist()) != {0}:
        raise ValueError("only FRONT camera 0 may enter training")
    if not np.any((camera_ids == 0) & is_validation):
        raise ValueError("FRONT interpolation validation frames are missing")
    for camera_id in (1, 2):
        if not np.all(is_validation[camera_ids == camera_id]):
            raise ValueError("side camera {} leaked into training".format(camera_id))

    counts = {}
    for camera_id in sorted(expected_cameras):
        mask = camera_ids == camera_id
        counts[str(camera_id)] = {
            "total": int(mask.sum()),
            "train": int((mask & ~is_validation).sum()),
            "test": int((mask & is_validation).sum()),
            "unique_times": int(np.unique(time_stamps[mask]).size),
        }
    return {"passed": True, "images": image_count, "cameras": counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_path")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = audit_cross_camera_scene(args.scene_path)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n")


if __name__ == "__main__":
    main()
