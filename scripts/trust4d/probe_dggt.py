#!/usr/bin/env python3
"""Record the released DGGT single-clip output contract."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_summary(value):
    value = value.detach()
    summary = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": value.numel(),
    }
    if value.numel() == 0:
        summary.update(finite_fraction=None, min=None, max=None)
        return summary

    if value.is_floating_point():
        finite = torch.isfinite(value)
        finite_values = value[finite]
        summary["finite_fraction"] = float(finite.float().mean().item())
        summary["min"] = (
            float(finite_values.min().item()) if finite_values.numel() else None
        )
        summary["max"] = (
            float(finite_values.max().item()) if finite_values.numel() else None
        )
    else:
        summary["finite_fraction"] = 1.0
        summary["min"] = float(value.min().item())
        summary["max"] = float(value.max().item())
    return summary


def summarize_tree(value):
    if torch.is_tensor(value):
        return tensor_summary(value)
    if isinstance(value, dict):
        return {key: summarize_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize_tree(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dggt-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dggt_root = args.dggt_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    image_paths = [path.expanduser().resolve() for path in args.images]
    sys.path.insert(0, str(dggt_root))

    from dggt.models.vggt import VGGT
    from dggt.utils.load_fn import load_and_preprocess_images

    if not torch.cuda.is_available():
        raise RuntimeError("EXP-001 requires an NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    model = VGGT().to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)

    with torch.inference_mode():
        warmup = model(images)
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - start

    required_keys = {
        "pose_enc",
        "world_points",
        "world_points_conf",
        "gs_map",
        "gs_conf",
        "dynamic_conf",
        "depth",
        "depth_conf",
    }
    missing_keys = sorted(required_keys - predictions.keys())
    required_tensors = [
        predictions[key]
        for key in sorted(required_keys - set(missing_keys))
        if torch.is_tensor(predictions[key])
    ]
    all_finite = all(
        not value.is_floating_point() or bool(torch.isfinite(value).all().item())
        for value in required_tensors
    )
    sequence_is_four = all(
        value.ndim >= 2 and value.shape[1] == 4 for value in required_tensors
    )
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024**2
    gate = {
        "missing_required_keys": missing_keys,
        "required_outputs_are_tensors": len(required_tensors) == len(required_keys),
        "required_outputs_all_finite": all_finite,
        "required_sequence_dimension_is_four": sequence_is_four,
        "peak_allocated_below_44_gib": peak_allocated_mib < 44 * 1024,
    }
    gate["passed"] = all(
        value for key, value in gate.items() if key != "missing_required_keys"
    ) and not missing_keys

    result = {
        "experiment_id": "EXP-001",
        "dggt_commit": subprocess.check_output(
            ["git", "-C", str(dggt_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
        },
        "images": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in image_paths
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input": tensor_summary(images),
        "outputs": summarize_tree(predictions),
        "wall_time_seconds": wall_time,
        "peak_memory_allocated_mib": peak_allocated_mib,
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "gate": gate,
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
