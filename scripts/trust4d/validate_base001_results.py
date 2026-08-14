#!/usr/bin/env python3
"""Validate the locked BASE-001 checkpoint and released AD-GS metrics."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ANCHORS = {
    "PSNR": 34.9363,
    "SSIM": 0.95216,
    "LPIPS(VGG)": 0.18436,
}
TOLERANCES = {
    "PSNR": 0.50,
    "SSIM": 0.010,
    "LPIPS(VGG)": 0.020,
}
REQUIRED_METRICS = ("PSNR", "SSIM", "LPIPS(VGG)", "LPIPS(ALEX)", "FPS")


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def read_metrics(path):
    with path.open() as stream:
        payload = json.load(stream)
    if "ours_60000" not in payload or not isinstance(payload["ours_60000"], dict):
        raise ValueError(f"{path.name} is missing ours_60000")
    metrics = payload["ours_60000"]
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"{path.name} is missing metrics: {missing}")
    checked = {}
    for name in REQUIRED_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path.name} metric {name} is not numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{path.name} metric {name} is not finite")
        checked[name] = value
    if checked["PSNR"] <= 0 or checked["FPS"] <= 0:
        raise ValueError(f"{path.name} requires positive PSNR and FPS")
    if not 0 <= checked["SSIM"] <= 1:
        raise ValueError(f"{path.name} SSIM is outside [0,1]")
    if checked["LPIPS(VGG)"] < 0 or checked["LPIPS(ALEX)"] < 0:
        raise ValueError(f"{path.name} LPIPS is negative")
    return checked


def validate_run(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    required_files = {
        "point_cloud": run_dir / "point_cloud/iteration_60000/point_cloud.ply",
        "environment_map": run_dir / "point_cloud/iteration_60000/env.pth",
        "test_results": run_dir / "results.json",
        "train_results": run_dir / "results-train.json",
        "config": run_dir / "cfg_args",
    }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"BASE-001 output is missing files: {missing}")
    empty = [name for name, path in required_files.items() if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"BASE-001 output has empty files: {empty}")

    test_metrics = read_metrics(required_files["test_results"])
    train_metrics = read_metrics(required_files["train_results"])
    deviations = {
        name: abs(test_metrics[name] - anchor) for name, anchor in ANCHORS.items()
    }
    failed = {
        name: {"deviation": deviations[name], "tolerance": TOLERANCES[name]}
        for name in ANCHORS
        if deviations[name] > TOLERANCES[name]
        and not math.isclose(
            deviations[name], TOLERANCES[name], rel_tol=0.0, abs_tol=1e-12
        )
    }
    if failed:
        raise ValueError(f"BASE-001 historical-anchor tolerance failed: {failed}")

    return {
        "experiment_id": "BASE-001",
        "passed": True,
        "run_dir": str(run_dir),
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "historical_anchors": ANCHORS,
        "tolerances": TOLERANCES,
        "absolute_deviations": deviations,
        "artifacts": {
            name: file_record(path) for name, path in required_files.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = validate_run(args.run_dir)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
