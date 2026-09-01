#!/usr/bin/env python3
"""Create a deterministic inventory of preprocessed AD-GS Waymo scenes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


SCENES = (
    "scene006",
    "scene026",
    "scene090",
    "scene105",
    "scene108",
    "scene134",
    "scene150",
    "scene181",
)
MODALITIES = ("image", "depth", "semantic", "sky", "flow")
CRITICAL_FILES = ("cameras.npz", "points3d.ply", "colmap.ply")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_directory(path):
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    layout = hashlib.sha256()
    total_bytes = 0
    for item in files:
        size = item.stat().st_size
        total_bytes += size
        record = "{}\0{}\n".format(item.relative_to(path).as_posix(), size)
        layout.update(record.encode("utf-8"))
    return {
        "files": len(files),
        "bytes": total_bytes,
        "layout_sha256": layout.hexdigest(),
    }


def inventory_scene(root, scene):
    scene_root = root / scene
    errors = []
    modalities = {}
    critical = {}

    if not scene_root.is_dir():
        return {
            "passed": False,
            "errors": ["missing scene directory: {}".format(scene)],
            "modalities": modalities,
            "critical_files": critical,
        }

    for name in MODALITIES:
        path = scene_root / name
        if not path.is_dir():
            errors.append("{}/{} is missing".format(scene, name))
            continue
        record = inventory_directory(path)
        modalities[name] = record
        if record["files"] == 0:
            errors.append("{}/{} contains no files".format(scene, name))

    image_count = modalities.get("image", {}).get("files")
    if image_count:
        for name in ("depth", "semantic", "sky"):
            count = modalities.get(name, {}).get("files")
            if count is not None and count != image_count:
                errors.append(
                    "{}/{} has {} files, expected {} to match image".format(
                        scene, name, count, image_count
                    )
                )

    for name in CRITICAL_FILES:
        path = scene_root / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append("{}/{} is missing or empty".format(scene, name))
            continue
        critical[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return {
        "passed": not errors,
        "errors": errors,
        "modalities": modalities,
        "critical_files": critical,
    }


def inventory_waymo(root, scenes=SCENES):
    root = Path(root)
    records = {scene: inventory_scene(root, scene) for scene in scenes}
    return {
        "schema": "adgs-waymo-preprocessed-inventory-v1",
        "passed": all(record["passed"] for record in records.values()),
        "scenes": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing scene006 ... scene181")
    parser.add_argument("--output", type=Path, help="optional JSON receipt path")
    args = parser.parse_args()

    receipt = inventory_waymo(args.root)
    serialized = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
