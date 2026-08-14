#!/usr/bin/env python3
"""Independently validate the locked EXP-001 evidence directory."""

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_COMMIT = "a3276d2bbe4cbb03bcc117830b1836110a27adeb"
EXPECTED_CHECKPOINT_SIZE_BYTES = 5_411_266_466
EXPECTED_PACKAGE_VERSIONS = {
    "torch": "2.4.1",
    "torchvision": "0.19.1",
    "gsplat": "1.5.3",
}
REQUIRED_OUTPUTS = {
    "pose_enc",
    "world_points",
    "world_points_conf",
    "gs_map",
    "gs_conf",
    "dynamic_conf",
    "depth",
    "depth_conf",
}
MINIMUM_RESERVED_MARGIN_MIB = 4096.0


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_number(payload, key, *, positive=False):
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{key} must be {'positive and ' if positive else ''}finite")
    return value


def _iter_tensor_summaries(value):
    if isinstance(value, dict):
        if {"shape", "dtype", "device", "numel", "finite_fraction"} <= value.keys():
            yield value
        else:
            for item in value.values():
                yield from _iter_tensor_summaries(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_tensor_summaries(item)


def _validate_tensor_summary(name, summary, *, require_sequence=False):
    shape = summary.get("shape")
    numel = summary.get("numel")
    if not isinstance(shape, list) or not all(isinstance(item, int) for item in shape):
        raise ValueError(f"{name} has an invalid shape summary")
    if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
        raise ValueError(f"{name} must be nonempty")
    if require_sequence and (len(shape) < 2 or shape[0] != 1 or shape[1] != 4):
        raise ValueError(f"{name} must have batch/sequence dimensions [1,4]")
    if str(summary.get("dtype", "")).startswith("torch.float"):
        finite_fraction = summary.get("finite_fraction")
        if finite_fraction != 1.0:
            raise ValueError(f"{name} is not entirely finite")


def validate_exp001(result_dir):
    result_dir = Path(result_dir).expanduser().resolve()
    required_files = [
        "metrics.json",
        "selection.json",
        "probe_exitcode.txt",
        "wall_time_seconds.txt",
        "git_remote.txt",
        "git_commit.txt",
        "git_status.txt",
        "nvidia-smi.txt",
        "environment.txt",
        "command.sh",
        "stdout.log",
        "stderr.log",
        "selection.stdout.log",
        "selection.stderr.log",
    ]
    missing = [name for name in required_files if not (result_dir / name).is_file()]
    if missing:
        raise ValueError(f"EXP-001 evidence is missing files: {missing}")
    if (result_dir / "probe_exitcode.txt").read_text().strip() != "0":
        raise ValueError("EXP-001 probe exit code is not zero")

    metrics_path = result_dir / "metrics.json"
    selection_path = result_dir / "selection.json"
    metrics = json.loads(metrics_path.read_text())
    selection = json.loads(selection_path.read_text())
    if metrics.get("experiment_id") != "EXP-001":
        raise ValueError("metrics experiment_id is not EXP-001")
    source = metrics.get("source", {})
    if source.get("commit") != EXPECTED_COMMIT or source.get("working_tree_clean") is not True:
        raise ValueError("DGGT source contract failed independent validation")

    checkpoint = metrics.get("checkpoint", {})
    if checkpoint.get("size_bytes") != EXPECTED_CHECKPOINT_SIZE_BYTES:
        raise ValueError("checkpoint size failed independent validation")
    checkpoint_sha = checkpoint.get("sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise ValueError("checkpoint SHA-256 is missing or malformed")

    packages = metrics.get("runtime_packages", {})
    for package, expected in EXPECTED_PACKAGE_VERSIONS.items():
        actual = packages.get(package)
        if not isinstance(actual, str) or actual.split("+", 1)[0] != expected:
            raise ValueError(f"runtime package mismatch for {package}: {actual}")
    if not isinstance(packages.get("scikit-learn"), str):
        raise ValueError("scikit-learn runtime record is missing")

    selection_contract = metrics.get("selection_manifest", {})
    if selection.get("count") != 4 or len(selection.get("images", [])) != 4:
        raise ValueError("selection manifest must contain exactly four images")
    if selection_contract.get("content") != selection:
        raise ValueError("metrics do not embed the exact selection manifest")
    if selection_contract.get("sha256") != file_sha256(selection_path):
        raise ValueError("selection manifest SHA-256 mismatch")

    environment = metrics.get("environment", {})
    if environment.get("gpu") != "NVIDIA A40":
        raise ValueError("EXP-001 did not record an NVIDIA A40")
    if environment.get("cuda_visible_devices") != "0":
        raise ValueError("EXP-001 was not isolated with CUDA_VISIBLE_DEVICES=0")
    if environment.get("compute_capability") != [8, 6]:
        raise ValueError("EXP-001 compute capability is not A40 sm_86")
    if environment.get("cuda_runtime") != "12.1":
        raise ValueError("EXP-001 PyTorch CUDA runtime is not 12.1")

    input_summary = metrics.get("input", {})
    _validate_tensor_summary("input", input_summary)
    input_shape = input_summary.get("shape")
    if (
        len(input_shape) != 5
        or input_shape[:3] != [1, 4, 3]
        or input_shape[3] <= 0
        or input_shape[4] <= 0
        or input_shape[3] % 14
        or input_shape[4] % 14
    ):
        raise ValueError("input tensor must be [1,4,3,H,W] with H/W divisible by 14")

    outputs = metrics.get("outputs", {})
    missing_outputs = sorted(REQUIRED_OUTPUTS - outputs.keys())
    if missing_outputs:
        raise ValueError(f"metrics are missing required outputs: {missing_outputs}")
    for name in REQUIRED_OUTPUTS:
        _validate_tensor_summary(name, outputs[name], require_sequence=True)
    summaries = list(_iter_tensor_summaries(outputs))
    if not summaries:
        raise ValueError("metrics contain no output tensor summaries")
    for index, summary in enumerate(summaries):
        if str(summary.get("dtype", "")).startswith("torch.float") and summary.get(
            "finite_fraction"
        ) != 1.0:
            raise ValueError(f"output tensor summary {index} is not entirely finite")

    total = _require_number(environment, "total_memory_mib", positive=True)
    allocated = _require_number(metrics, "peak_memory_allocated_mib", positive=True)
    reserved = _require_number(metrics, "peak_memory_reserved_mib", positive=True)
    margin = _require_number(metrics, "reserved_memory_margin_mib", positive=True)
    if allocated > reserved:
        raise ValueError("peak allocated memory exceeds peak reserved memory")
    if not math.isclose(total - reserved, margin, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("recorded reserved-memory margin is inconsistent")
    if margin < MINIMUM_RESERVED_MARGIN_MIB:
        raise ValueError("reserved-memory margin is below 4096 MiB")
    _require_number(metrics, "wall_time_seconds", positive=True)
    model_parameters = metrics.get("model_parameters")
    if isinstance(model_parameters, bool) or not isinstance(model_parameters, int) or model_parameters <= 0:
        raise ValueError("model_parameters must be a positive integer")

    gate = metrics.get("gate", {})
    expected_gate_fields = {
        "required_outputs_are_tensors",
        "required_outputs_nonempty",
        "all_output_tensors_finite",
        "required_sequence_dimension_matches",
        "peak_allocated_not_above_reserved",
        "reserved_memory_margin_at_least_4096_mib",
        "passed",
    }
    if gate.get("missing_required_keys") != [] or any(
        gate.get(field) is not True for field in expected_gate_fields
    ):
        raise ValueError("probe gate did not pass every locked condition")

    return {
        "experiment_id": "EXP-001",
        "passed": True,
        "result_dir": str(result_dir),
        "dggt_commit": source["commit"],
        "selection_sha256": file_sha256(selection_path),
        "checkpoint_sha256": checkpoint_sha,
        "input_shape": input_shape,
        "output_keys": sorted(outputs),
        "wall_time_seconds": metrics["wall_time_seconds"],
        "peak_memory_allocated_mib": allocated,
        "peak_memory_reserved_mib": reserved,
        "reserved_memory_margin_mib": margin,
        "metrics_sha256": file_sha256(metrics_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = validate_exp001(args.result_dir)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
