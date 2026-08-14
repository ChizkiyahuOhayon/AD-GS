#!/usr/bin/env python3
"""Fail-closed smoke test for the pinned AD-GS baseline runtime."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path


EXPECTED_VERSIONS = {
    "torch": "1.13.1",
    "torchvision": "0.14.1",
    "pytorch3d": "0.7.4",
    "numpy": "1.21.6",
}


def check_version_contract(actual):
    mismatches = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in EXPECTED_VERSIONS.items()
        if actual.get(name, "").split("+", 1)[0] != expected
    }
    if mismatches:
        raise ValueError(f"AD-GS runtime version mismatch: {mismatches}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adgs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    adgs_root = args.adgs_root.expanduser().resolve()
    sys.path.insert(0, str(adgs_root))

    import numpy as np
    import torch

    actual = {
        name: importlib.metadata.version(name)
        for name in EXPECTED_VERSIONS
    }
    check_version_contract(actual)
    if torch.version.cuda != "11.7":
        raise RuntimeError(f"expected PyTorch CUDA 11.7, got {torch.version.cuda}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_name(0) != "NVIDIA A40":
        raise RuntimeError(f"expected NVIDIA A40, got {torch.cuda.get_device_name(0)}")
    if torch.cuda.get_device_capability(0) != (8, 6):
        raise RuntimeError(
            f"expected compute capability 8.6, got {torch.cuda.get_device_capability(0)}"
        )

    from diff_gaussian_rasterization import GaussianRasterizer
    from pytorch3d.ops import knn_points
    from simple_knn._C import distCUDA2

    first = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device="cuda")
    second = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device="cuda")
    knn = knn_points(first, second, K=1)
    if knn.idx.tolist() != [[[0], [1]]] or not torch.isfinite(knn.dists).all():
        raise RuntimeError("pytorch3d.ops.knn_points smoke test failed")

    points = torch.rand((32, 3), device="cuda", dtype=torch.float32)
    distances = distCUDA2(points)
    if distances.shape != (32,) or not torch.isfinite(distances).all():
        raise RuntimeError("simple_knn.distCUDA2 smoke test failed")
    if bool((distances < 0).any().item()):
        raise RuntimeError("simple_knn.distCUDA2 returned a negative squared distance")

    import render  # noqa: F401
    import train  # noqa: F401

    result = {
        "passed": True,
        "versions": actual,
        "python": sys.version,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "knn_indices": knn.idx.cpu().tolist(),
        "knn_distances": knn.dists.cpu().tolist(),
        "simple_knn_min": float(distances.min().item()),
        "simple_knn_max": float(distances.max().item()),
        "rasterizer_class": GaussianRasterizer.__name__,
        "numpy": np.__version__,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
