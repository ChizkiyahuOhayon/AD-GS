#!/usr/bin/env python3
"""Freeze train-only EXP-002 interventions and semantic-mask query pixels."""

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .audit_official_source import audit_repository
    from .download_waymo_manifest import load_manifest, sha256_file, write_json
    from .validate_waymo_scene006 import validate_scene
except ImportError:
    from audit_official_source import audit_repository
    from download_waymo_manifest import load_manifest, sha256_file, write_json
    from validate_waymo_scene006 import validate_scene


EXPECTED_SCENES = ("scene006", "scene026", "scene090")
ANCHORS = (5, 25, 45, 65)
TARGET_WIDTH = 518
PATCH_DIVISOR = 14
MAX_CROP_HEIGHT = 518
EROSION_ITERATIONS = 5
GRID_STRIDE = 16
GRID_ORIGIN = 8
MAX_QUERIES = 128
MIN_QUERIES = 16


def dggt_crop_geometry(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("source image dimensions must be positive")
    resized_width = TARGET_WIDTH
    resized_height = round(height * (resized_width / width) / PATCH_DIVISOR) * PATCH_DIVISOR
    if resized_height <= 0:
        raise ValueError("DGGT resize produced a nonpositive height")
    crop_top = 0
    output_height = resized_height
    if resized_height > MAX_CROP_HEIGHT:
        crop_top = (resized_height - MAX_CROP_HEIGHT) // 2
        output_height = MAX_CROP_HEIGHT
    return {
        "source_width": width,
        "source_height": height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "crop_top": crop_top,
        "crop_left": 0,
        "output_width": resized_width,
        "output_height": output_height,
    }


def transform_object_mask(mask, geometry):
    if mask.ndim != 2:
        raise ValueError("semantic object mask must be two-dimensional")
    expected_shape = (geometry["source_height"], geometry["source_width"])
    if mask.shape != expected_shape:
        raise ValueError(f"semantic mask shape must be {expected_shape}, got {mask.shape}")
    image = Image.fromarray((mask.astype(np.bool_) * 255).astype(np.uint8))
    image = image.resize(
        (geometry["resized_width"], geometry["resized_height"]),
        Image.Resampling.NEAREST,
    )
    transformed = np.asarray(image, dtype=np.uint8) > 0
    if geometry["resized_height"] > MAX_CROP_HEIGHT:
        top = geometry["crop_top"]
        transformed = transformed[top : top + MAX_CROP_HEIGHT]
    expected_output = (geometry["output_height"], geometry["output_width"])
    if transformed.shape != expected_output:
        raise RuntimeError(
            f"transformed mask shape mismatch: {transformed.shape} != {expected_output}"
        )
    return transformed


def binary_erode_full_neighborhood(mask, iterations=EROSION_ITERATIONS):
    if mask.ndim != 2 or iterations < 0:
        raise ValueError("erosion expects a 2D mask and nonnegative iterations")
    eroded = mask.astype(np.bool_, copy=True)
    for _ in range(iterations):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        height, width = eroded.shape
        neighborhoods = [
            padded[dy : dy + height, dx : dx + width]
            for dy in range(3)
            for dx in range(3)
        ]
        eroded = np.logical_and.reduce(neighborhoods)
    return eroded


def select_grid_queries(eroded_mask):
    if eroded_mask.ndim != 2:
        raise ValueError("query selection expects a two-dimensional mask")
    height, width = eroded_mask.shape
    candidates = [
        [int(x), int(y)]
        for y in range(GRID_ORIGIN, height, GRID_STRIDE)
        for x in range(GRID_ORIGIN, width, GRID_STRIDE)
        if eroded_mask[y, x]
    ]
    if len(candidates) > MAX_QUERIES:
        ranks = np.rint(np.linspace(0, len(candidates) - 1, MAX_QUERIES)).astype(
            np.int64
        )
        if len(np.unique(ranks)) != MAX_QUERIES:
            raise RuntimeError("equally spaced query ranks are not unique")
        selected = [candidates[int(rank)] for rank in ranks]
        selected_ranks = ranks.tolist()
    else:
        selected = candidates
        selected_ranks = list(range(len(candidates)))
    return candidates, selected, selected_ranks


def array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def intervention_indices(anchor):
    original = [anchor, anchor + 5, anchor + 10, anchor + 14]
    return {
        "original": original,
        "reverse": list(reversed(original)),
        "sparse": [anchor, anchor + 10, anchor + 14],
        "interior_shifted": [anchor, anchor + 4, anchor + 9, anchor + 14],
    }


def build_query_manifest(scene_paths):
    by_name = {}
    for value in scene_paths:
        path = Path(value).expanduser().resolve()
        if path.name in by_name:
            raise ValueError(f"duplicate scene path for {path.name}")
        by_name[path.name] = path
    if tuple(sorted(by_name)) != tuple(sorted(EXPECTED_SCENES)):
        raise ValueError(f"EXP-002 requires exactly these scenes: {EXPECTED_SCENES}")

    official_manifest = load_manifest()
    official_by_scene = {item["scene"]: item for item in official_manifest["sequences"]}
    scene_results = []
    total_windows = 0
    supported_windows = 0
    for scene_name in EXPECTED_SCENES:
        scene = by_name[scene_name]
        sequence = official_by_scene[scene_name]
        frame_count = sequence["last_frame"] - sequence["first_frame"] + 1
        base = validate_scene(scene, frame_count)
        images = {index: record for index, record in enumerate(base["images"])}
        with np.load(scene / "cameras.npz", allow_pickle=False) as cameras:
            is_val = np.asarray(cameras["is_val_list"], dtype=np.bool_)
            timestamps = np.asarray(cameras["time_stamps"])

        windows = []
        for anchor in ANCHORS:
            interventions = intervention_indices(anchor)
            referenced_indices = sorted(
                {index for values in interventions.values() for index in values}
            )
            if referenced_indices[-1] >= frame_count:
                raise ValueError(f"{scene_name} anchor {anchor} exceeds frame count")
            held_out = [index for index in referenced_indices if bool(is_val[index])]
            if held_out:
                raise ValueError(
                    f"{scene_name} anchor {anchor} uses held-out indices: {held_out}"
                )

            image_record = images[anchor]
            geometry = dggt_crop_geometry(
                image_record["width"], image_record["height"]
            )
            semantic_path = scene / "semantic" / f"mask_{anchor:06d}.npy"
            if not semantic_path.is_file():
                raise ValueError(f"missing anchor semantic mask: {semantic_path}")
            semantic = np.load(semantic_path, allow_pickle=False)
            if not np.issubdtype(semantic.dtype, np.integer):
                raise ValueError(f"semantic mask must have integer dtype: {semantic_path}")
            if not np.isfinite(semantic).all() or semantic.min() < 0:
                raise ValueError(f"semantic mask is invalid: {semantic_path}")
            transformed = transform_object_mask(semantic > 0, geometry)
            eroded = binary_erode_full_neighborhood(transformed)
            candidates, selected, selected_ranks = select_grid_queries(eroded)
            supported = len(selected) >= MIN_QUERIES
            total_windows += 1
            supported_windows += int(supported)
            windows.append(
                {
                    "anchor": anchor,
                    "anchor_timestamp": float(timestamps[anchor]),
                    "interventions": interventions,
                    "referenced_images": [
                        {
                            "index": index,
                            "timestamp": float(timestamps[index]),
                            "path": images[index]["path"],
                            "size_bytes": images[index]["size_bytes"],
                            "sha256": images[index]["sha256"],
                        }
                        for index in referenced_indices
                    ],
                    "semantic_mask": {
                        "path": str(semantic_path.resolve()),
                        "size_bytes": semantic_path.stat().st_size,
                        "sha256": sha256_file(semantic_path),
                        "dtype": str(semantic.dtype),
                        "shape": list(semantic.shape),
                        "positive_pixels": int(np.count_nonzero(semantic > 0)),
                    },
                    "crop_geometry": geometry,
                    "transformed_object_mask_sha256": array_sha256(transformed),
                    "eroded_object_mask_sha256": array_sha256(eroded),
                    "grid_candidate_count": len(candidates),
                    "selected_rank_indices": selected_ranks,
                    "queries_xy": selected,
                    "selected_query_count": len(selected),
                    "valid_query_support": supported,
                    "coverage_failure_reason": (
                        None if supported else f"fewer_than_{MIN_QUERIES}_queries"
                    ),
                }
            )
        scene_results.append(
            {
                "scene": scene_name,
                "path": str(scene),
                "frame_count": frame_count,
                "cameras": base["cameras"],
                "validation_indices": base["validation_indices"],
                "windows": windows,
            }
        )

    return {
        "experiment_id": "EXP-002",
        "stage": "train_only_query_manifest",
        "leakage_boundary": (
            "Grounded-SAM semantic priors and train RGB/cameras only; no Waymo "
            "boxes, IDs, categories, velocities, held-out RGB, or evaluation metrics"
        ),
        "algorithm": {
            "resize_mode": "DGGT crop: width 518, aspect height rounded to multiple of 14",
            "mask_interpolation": "PIL nearest-neighbor",
            "max_crop_height": MAX_CROP_HEIGHT,
            "erosion": "3x3 full binary neighborhood, false padding, 5 iterations",
            "grid_stride": GRID_STRIDE,
            "grid_origin_xy": [GRID_ORIGIN, GRID_ORIGIN],
            "candidate_order": "row-major (y,x)",
            "max_queries": MAX_QUERIES,
            "subsampling": (
                "numpy.rint(linspace(0,N-1,128)); round-to-nearest-even"
            ),
            "minimum_supported_queries": MIN_QUERIES,
        },
        "anchors": list(ANCHORS),
        "total_preselected_windows": total_windows,
        "query_supported_windows": supported_windows,
        "scenes": scene_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if output == repository or repository in output.parents:
        raise ValueError("query manifest output must be outside the Git repository")
    if output.exists():
        raise ValueError(f"query manifest output directory must be new: {output}")
    source_audit = audit_repository(repository)
    result = build_query_manifest(args.scene)

    output.mkdir(parents=True)
    manifest_path = output / "query-manifest.json"
    write_json(manifest_path, result)
    write_json(output / "official_source_audit.json", source_audit)
    (output / "git_commit.txt").write_text(source_audit["head"] + "\n")
    (output / "command.sh").write_text(
        shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        + "\n"
    )
    (output / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  query-manifest.json\n"
    )
    artifact_lines = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifacts.sha256":
            artifact_lines.append(f"{sha256_file(path)}  {path.name}")
    (output / "artifacts.sha256").write_text("\n".join(artifact_lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
