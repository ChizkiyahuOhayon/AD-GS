#!/usr/bin/env python3
"""Fail-closed runtime check for the single-A40 GF-DGS environment."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "python": "3.8",
    "torch": "2.0.1",
    "torch_cuda": "11.8",
    "numpy": "1.23.5",
    "pytorch3d": "0.7.4",
    "roma": "1.5.1",
    "setuptools": "69.5.1",
}


def validate_runtime(record):
    errors = []
    for name, expected in EXPECTED.items():
        if record.get(name) != expected:
            errors.append(
                "{} is {}, expected {}".format(name, record.get(name), expected)
            )
    if "A40" not in record.get("gpu_name", ""):
        errors.append("GPU is {}, expected NVIDIA A40".format(record.get("gpu_name")))
    if record.get("compute_capability") != [8, 6]:
        errors.append(
            "compute capability is {}, expected [8, 6]".format(
                record.get("compute_capability")
            )
        )
    return errors


def git_value(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo)] + list(args), text=True
    ).strip()


def audit_repository(repo, expected_commit):
    errors = []
    try:
        head = git_value(repo, "rev-parse", "HEAD")
        status = git_value(repo, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"passed": False, "errors": [str(exc)]}
    if head != expected_commit:
        errors.append("HEAD is {}, expected {}".format(head, expected_commit))
    if status:
        errors.append("tracked worktree is dirty: {}".format(status.splitlines()))
    return {"passed": not errors, "head": head, "status": status, "errors": errors}


def package_version(name):
    return importlib.metadata.version(name)


def check_cuda_runtime():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    device = torch.device("cuda:0")
    probe = torch.arange(4, dtype=torch.float32, device=device)
    if float(probe.square().sum().item()) != 14.0:
        raise RuntimeError("CUDA allocation probe returned an invalid value")
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": "{}.{}".format(sys.version_info.major, sys.version_info.minor),
        "torch": torch.__version__.split("+")[0],
        "torch_cuda": torch.version.cuda,
        "numpy": package_version("numpy"),
        "pytorch3d": package_version("pytorch3d"),
        "roma": package_version("roma"),
        "setuptools": package_version("setuptools"),
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
    }


def check_cuda_extensions():
    import torch
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from pytorch3d.ops import knn_points
    from simple_knn._C import distCUDA2

    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device="cuda:0",
    )
    distances = distCUDA2(points)
    if distances.shape != (4,) or not bool(torch.isfinite(distances).all()):
        raise RuntimeError("simple-knn CUDA probe returned invalid distances")

    settings = GaussianRasterizationSettings(
        image_height=8,
        image_width=8,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device="cuda:0"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda:0"),
        projmatrix=torch.eye(4, device="cuda:0"),
        sh_degree=0,
        campos=torch.zeros(3, device="cuda:0"),
        prefiltered=False,
        inv_depth=False,
        debug=False,
    )
    visible = GaussianRasterizer(settings).markVisible(points)
    if visible.shape != (4,) or visible.dtype != torch.bool:
        raise RuntimeError("rasterizer CUDA probe returned an invalid mask")

    result = knn_points(points[None], points[None], K=1)
    if result.dists.shape != (1, 4, 1) or not bool(torch.isfinite(result.dists).all()):
        raise RuntimeError("PyTorch3D CUDA probe returned invalid distances")
    torch.cuda.synchronize()
    return {
        "simple_knn": True,
        "diff_gaussian_rasterization": True,
        "pytorch3d_knn": True,
    }


def run_audit(repo, expected_commit):
    repository = audit_repository(repo, expected_commit)
    errors = list(repository["errors"])
    runtime = {}
    extensions = {}
    try:
        runtime = check_cuda_runtime()
        errors.extend(validate_runtime(runtime))
    except Exception as exc:
        errors.append("runtime probe failed: {}".format(exc))
    try:
        extensions = check_cuda_extensions()
    except Exception as exc:
        errors.append("CUDA extension probe failed: {}".format(exc))
    return {
        "schema": "gfdgs-a40-environment-v1",
        "passed": not errors,
        "errors": errors,
        "repository": repository,
        "runtime": runtime,
        "extensions": extensions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    receipt = run_audit(args.repo, args.expected_commit)
    serialized = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
