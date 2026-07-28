#!/usr/bin/env python3
"""Audit whether the scale--range orbit is realizable by AD-GS motion bases.

AD-GS represents each dynamic Gaussian trajectory with a constant canonical
position, a uniform B-spline, and Fourier terms.  The camera-relative gauge is
therefore exact in host parameter space only when the camera-center trajectory
is in the same sampled basis span.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


WAYMO_SCENE_FRAMES = {
    "scene006": 86,
    "scene026": 101,
    "scene090": 103,
    "scene105": 167,
    "scene108": 96,
    "scene134": 93,
    "scene150": 102,
    "scene181": 161,
}


def deboor_cox_matrix(order):
    if order == 0:
        return np.array([[1.0]], dtype=np.float64)
    prior = deboor_cox_matrix(order - 1)
    prior_left = np.concatenate((prior, np.zeros((1, prior.shape[1]))), axis=0)
    prior_right = np.concatenate((np.zeros((1, prior.shape[1])), prior), axis=0)
    left = np.zeros((order, order + 1), dtype=np.float64)
    right = np.zeros((order, order + 1), dtype=np.float64)
    indices = np.arange(order)
    left[indices, indices] = indices + 1
    left[indices, indices + 1] = order - indices - 1
    right[indices, indices] = -1
    right[indices, indices + 1] = 1
    return (prior_left @ left + prior_right @ right) / order


def adgs_design_matrix(times, frame_count, downsample_ratio=3, order=5, fft_order=6):
    control_count = frame_count // downsample_ratio
    order = min(order, control_count - 1)
    if control_count <= order:
        raise ValueError("The B-spline needs more control points than its order.")

    interval_count = control_count - order
    local_matrix = deboor_cox_matrix(order)
    bspline = np.zeros((len(times), control_count), dtype=np.float64)
    for row, time in enumerate(times):
        start = min(int(time * interval_count), interval_count - 1)
        local_time = time * interval_count - start
        powers = local_time ** np.arange(order + 1, dtype=np.float64)
        bspline[row, start : start + order + 1] = powers @ local_matrix

    frequencies = np.arange(1, fft_order + 1, dtype=np.float64) * math.pi
    fourier = np.concatenate(
        (np.sin(times[:, None] * frequencies), np.cos(times[:, None] * frequencies)),
        axis=1,
    )
    design = np.concatenate((np.ones((len(times), 1)), bspline, fourier), axis=1)
    return design, {
        "canonical_constant": 1,
        "bspline_control_count": control_count,
        "bspline_order": order,
        "fft_order": fft_order,
        "raw_parameter_count_per_coordinate": int(design.shape[1]),
    }


def fit_camera_basis(times, centers, train_mask, frame_count, gauge_scale):
    design, basis = adgs_design_matrix(times, frame_count)
    train_design = design[train_mask]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        train_design, centers[train_mask], rcond=1e-12
    )
    fitted = design @ coefficients
    residual = centers - fitted
    gauge_residual = abs(1.0 - gauge_scale) * residual

    def summarize(values, mask):
        distances = np.linalg.norm(values[mask], axis=1)
        return {
            "rms_m": float(np.sqrt(np.mean(distances**2))),
            "median_m": float(np.median(distances)),
            "max_m": float(np.max(distances)),
        }

    val_mask = np.logical_not(train_mask)
    result = {
        "frame_count": int(frame_count),
        "training_observations": int(train_mask.sum()),
        "held_out_observations": int(val_mask.sum()),
        "basis": basis,
        "sampled_training_rank": int(rank),
        "smallest_retained_singular_value": float(singular_values[rank - 1]),
        "camera_fit_train": summarize(residual, train_mask),
        "gauge_realization_train": summarize(gauge_residual, train_mask),
    }
    if bool(val_mask.any()):
        result["camera_fit_held_out"] = summarize(residual, val_mask)
        result["gauge_realization_held_out"] = summarize(gauge_residual, val_mask)
    return result


def waymo_train_mask(frame_count):
    mask = np.ones(frame_count, dtype=bool)
    mask[np.arange(4, frame_count, 4)] = False
    return mask


def synthetic_paths(frame_count):
    times = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    straight = np.stack(
        (30.0 * times, 2.0 * times**2, 0.1 * times), axis=1
    )
    angle = 0.65 * times
    smooth_turn = np.stack(
        (
            50.0 * np.sin(angle),
            50.0 * (1.0 - np.cos(angle)),
            0.5 * np.sin(math.pi * times),
        ),
        axis=1,
    )
    unmodeled_jitter = smooth_turn + np.stack(
        (
            0.03 * np.sin(17.0 * math.pi * times),
            0.02 * np.cos(19.0 * math.pi * times),
            0.01 * np.sin(23.0 * math.pi * times),
        ),
        axis=1,
    )
    return times, {
        "polynomial_ego_motion": straight,
        "smooth_turn": smooth_turn,
        "unmodeled_high_frequency_motion": unmodeled_jitter,
    }


def camera_centers_from_npz(path):
    metadata = np.load(path, allow_pickle=True)
    rotations = metadata["R"]
    translations = metadata["T"]
    timestamps = metadata["time_stamps"].astype(np.float64)
    train_mask = np.logical_not(metadata["is_val_list"].astype(bool))
    transforms = np.repeat(np.eye(4)[None], len(rotations), axis=0)
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3] = translations
    centers = np.linalg.inv(transforms)[:, :3, 3]
    times = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    frame_count = len(np.unique(timestamps))
    return times, centers, train_mask, frame_count


def run(gauge_scale, camera_files):
    synthetic = {}
    for scene, frame_count in WAYMO_SCENE_FRAMES.items():
        times, paths = synthetic_paths(frame_count)
        train_mask = waymo_train_mask(frame_count)
        synthetic[scene] = {
            name: fit_camera_basis(
                times, centers, train_mask, frame_count, gauge_scale
            )
            for name, centers in paths.items()
        }

    actual = {}
    for camera_file in camera_files:
        times, centers, train_mask, frame_count = camera_centers_from_npz(camera_file)
        actual[str(camera_file)] = fit_camera_basis(
            times, centers, train_mask, frame_count, gauge_scale
        )

    polynomial_max = max(
        item["polynomial_ego_motion"]["camera_fit_train"]["max_m"]
        for item in synthetic.values()
    )
    jitter_max = max(
        item["unmodeled_high_frequency_motion"]["camera_fit_train"]["rms_m"]
        for item in synthetic.values()
    )
    return {
        "host_position_model": (
            "canonical constant + uniform degree-5 B-spline with floor(N/3) "
            "controls + six sine/cosine frequencies"
        ),
        "gauge_scale": gauge_scale,
        "closure_condition": "camera centers at training times belong to col(B_AD-GS)",
        "realization_residual": "abs(1-lambda) * (I-P_B) * camera_center",
        "synthetic": synthetic,
        "actual_camera_files": actual,
        "gate": {
            "basis_reproduces_polynomial_motion": polynomial_max <= 1e-9,
            "arbitrary_camera_motion_is_not_exactly_closed": jitter_max >= 1e-5,
            "actual_waymo_closure_audited": bool(actual),
        },
        "conclusion": (
            "The orbit is exact in sampled AD-GS parameter space only under the "
            "camera-basis closure condition. Otherwise it is an approximate gauge, "
            "with a directly measurable residual on cameras.npz."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-scale", type=float, default=2.0)
    parser.add_argument("--cameras-npz", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run(args.lambda_scale, args.cameras_npz)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    required = (
        result["gate"]["basis_reproduces_polynomial_motion"],
        result["gate"]["arbitrary_camera_motion_is_not_exactly_closed"],
    )
    if not all(required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
