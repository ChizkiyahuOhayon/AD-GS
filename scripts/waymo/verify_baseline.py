#!/usr/bin/env python3
"""Audit a completed pristine AD-GS Waymo baseline run."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "analysis" / "manifests" / "adgs_waymo_baseline_protocol.json"


def load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def git_value(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo)] + list(args), text=True
    ).strip()


def audit_repository(repo, expected_commit, expected_origin):
    errors = []
    try:
        head = git_value(repo, "rev-parse", "HEAD")
        status = git_value(repo, "status", "--porcelain")
        origin = git_value(repo, "remote", "get-url", "origin")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"passed": False, "errors": [str(exc)]}
    if head != expected_commit:
        errors.append("HEAD is {}, expected {}".format(head, expected_commit))
    if status:
        errors.append("pristine repository is dirty: {}".format(status.splitlines()))
    if origin != expected_origin:
        errors.append("origin is {}, expected {}".format(origin, expected_origin))
    return {
        "passed": not errors,
        "head": head,
        "origin": origin,
        "status": status,
        "errors": errors,
    }


def require_file(path, errors):
    if not path.is_file() or path.stat().st_size == 0:
        errors.append("missing or empty file: {}".format(path))


def load_metric_record(path, iteration, required_metrics, errors):
    initial_error_count = len(errors)
    require_file(path, errors)
    if errors and (not path.is_file() or path.stat().st_size == 0):
        return None
    try:
        record = load_json(path)["ours_{}".format(iteration)]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append("invalid metric file {}: {}".format(path, exc))
        return None
    values = {}
    for name in required_metrics:
        value = record.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append("metric {} is missing or non-finite in {}".format(name, path))
        else:
            values[name] = float(value)
    return values if len(errors) == initial_error_count else None


def png_count(path):
    return len(list(path.glob("*.png"))) if path.is_dir() else 0


def audit_scene(protocol, scene, source_root, output_root, log_root):
    errors = []
    iteration = protocol["iterations"]
    required_metrics = protocol["hard_gate"]["required_metrics"]
    source = Path(source_root) / scene
    output = Path(output_root) / scene
    logs = Path(log_root)

    meta_path = source / "cameras.npz"
    require_file(meta_path, errors)
    expected_train = expected_test = None
    if meta_path.is_file() and meta_path.stat().st_size:
        try:
            with np.load(str(meta_path), allow_pickle=False) as meta:
                is_val = np.asarray(meta["is_val_list"], dtype=bool).reshape(-1)
            expected_test = int(is_val.sum())
            expected_train = int(is_val.size - expected_test)
            if expected_train == 0 or expected_test == 0:
                errors.append("empty train or test split in {}".format(meta_path))
        except (KeyError, OSError, ValueError) as exc:
            errors.append("invalid cameras.npz for {}: {}".format(scene, exc))

    for relative in ("cfg_args", "cameras.json", "input.ply"):
        require_file(output / relative, errors)
    require_file(output / "point_cloud" / "iteration_{}".format(iteration) / "point_cloud.ply", errors)
    require_file(output / "point_cloud" / "iteration_{}".format(iteration) / "env.pth", errors)

    train_log = logs / "{}.train.log".format(scene)
    render_log = logs / "{}.render.log".format(scene)
    train_exit = logs / "{}.train.exitcode".format(scene)
    render_exit = logs / "{}.render.exitcode".format(scene)
    for path in (train_log, render_log, train_exit, render_exit):
        require_file(path, errors)
    if train_log.is_file() and "Training complete." not in train_log.read_text(errors="replace"):
        errors.append("training completion marker missing: {}".format(train_log))
    if render_log.is_file() and "LPIPS(VGG)" not in render_log.read_text(errors="replace"):
        errors.append("render metric marker missing: {}".format(render_log))
    for path in (train_exit, render_exit):
        if path.is_file() and path.read_text().strip() != "0":
            errors.append("non-zero or invalid exit code: {}".format(path))

    split_counts = {}
    for split, expected in (("test", expected_test), ("train", expected_train)):
        base = output / split / "ours_{}".format(iteration)
        render_count = png_count(base / "renders")
        gt_count = png_count(base / "gt")
        split_counts[split] = {"expected": expected, "renders": render_count, "gt": gt_count}
        if expected is not None and (render_count != expected or gt_count != expected):
            errors.append(
                "{} frame count mismatch for {}: expected {}, renders {}, gt {}".format(
                    split, scene, expected, render_count, gt_count
                )
            )

    test_metrics = load_metric_record(
        output / "results.json", iteration, required_metrics, errors
    )
    train_metrics = load_metric_record(
        output / "results-train.json", iteration, required_metrics, errors
    )
    cfg_path = output / "cfg_args"
    if cfg_path.is_file() and "iterations={}".format(iteration) not in cfg_path.read_text(errors="replace"):
        errors.append("cfg_args does not lock iterations={}: {}".format(iteration, cfg_path))

    return {
        "scene": scene,
        "passed": not errors,
        "errors": errors,
        "split_counts": split_counts,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
    }


def aggregate_and_compare(protocol, scenes):
    names = protocol["hard_gate"]["required_metrics"]
    complete = [
        item
        for item in scenes
        if item["test_metrics"] is not None
        and isinstance(item["split_counts"]["test"]["expected"], int)
    ]
    if len(complete) != len(scenes):
        return {"passed": False, "errors": ["not all scenes have readable test metrics"]}
    macro = {name: float(np.mean([item["test_metrics"][name] for item in complete])) for name in names}
    weights = np.asarray([item["split_counts"]["test"]["expected"] for item in complete], dtype=float)
    weighted = {
        name: float(np.average([item["test_metrics"][name] for item in complete], weights=weights))
        for name in names
    }
    reference = protocol["published_reference"]["eight_scene_overall_metrics"]
    tolerances = protocol["comparison_policy"]["investigate_if_absolute_delta_exceeds"]
    comparisons = {}
    errors = []
    for name, target in reference.items():
        delta = macro[name] - target
        within = abs(delta) <= tolerances[name]
        comparisons[name] = {
            "macro": macro[name],
            "published": target,
            "delta": delta,
            "tolerance": tolerances[name],
            "within_tolerance": within,
        }
        if not within:
            errors.append("{} macro delta {} exceeds tolerance {}".format(name, delta, tolerances[name]))
    return {
        "passed": not errors,
        "errors": errors,
        "macro_scene_mean": macro,
        "frame_weighted_mean": weighted,
        "published_comparison": comparisons,
    }


def run_audit(protocol, baseline_repo, source_root, output_root, log_root):
    repository = audit_repository(
        baseline_repo, protocol["upstream_commit"], protocol["upstream_repository"]
    )
    scenes = [
        audit_scene(protocol, scene, source_root, output_root, log_root)
        for scene in protocol["scenes"]
    ]
    aggregate = aggregate_and_compare(protocol, scenes)
    passed = repository["passed"] and all(item["passed"] for item in scenes) and aggregate["passed"]
    return {
        "protocol": protocol["protocol"],
        "passed": passed,
        "repository": repository,
        "scenes": scenes,
        "aggregate": aggregate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = run_audit(
        load_json(args.protocol), args.baseline_repo, args.source_root, args.output_root, args.log_root
    )
    serialized = json.dumps(receipt, indent=2, sort_keys=True)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(serialized + "\n")
    print(serialized)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
