#!/usr/bin/env python3
"""Fit fixed-ID actor trajectories with a planar rigid pose plus z translation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def fit_planar_rigid(reference_xyz, target_xyz, weights, epsilon=1e-8):
    """Fit ``target = Rz(yaw) reference + translation`` by weighted Procrustes."""
    if reference_xyz.ndim != 2 or reference_xyz.shape[-1] != 3:
        raise ValueError("reference_xyz must have shape [N, 3]")
    if target_xyz.shape != reference_xyz.shape:
        raise ValueError("target_xyz must match reference_xyz")
    if weights.shape != reference_xyz.shape[:1]:
        raise ValueError("weights must have shape [N]")
    if torch.any(weights < 0) or not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must be finite and non-negative")

    total_weight = weights.sum()
    if float(total_weight) <= epsilon:
        raise ValueError("positive total weight is required")
    normalized_weights = weights / total_weight
    reference_center = torch.sum(normalized_weights[:, None] * reference_xyz, dim=0)
    target_center = torch.sum(normalized_weights[:, None] * target_xyz, dim=0)
    reference_xy = reference_xyz[:, :2] - reference_center[:2]
    target_xy = target_xyz[:, :2] - target_center[:2]

    cosine_term = torch.sum(
        normalized_weights
        * (reference_xy[:, 0] * target_xy[:, 0] + reference_xy[:, 1] * target_xy[:, 1])
    )
    sine_term = torch.sum(
        normalized_weights
        * (reference_xy[:, 0] * target_xy[:, 1] - reference_xy[:, 1] * target_xy[:, 0])
    )
    if float(torch.sqrt(cosine_term.square() + sine_term.square())) <= epsilon:
        raise ValueError("horizontal actor support is degenerate")

    yaw = torch.atan2(sine_term, cosine_term)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    rotated = torch.stack(
        (
            cosine * reference_xyz[:, 0] - sine * reference_xyz[:, 1],
            sine * reference_xyz[:, 0] + cosine * reference_xyz[:, 1],
            reference_xyz[:, 2],
        ),
        dim=-1,
    )
    translation = target_center - torch.sum(
        normalized_weights[:, None] * rotated, dim=0
    )
    predicted = rotated + translation
    return translation, yaw, predicted


def fit_actor_rigid_trajectories(
    world_xyz, actor_ids, weights=None, min_points=3, epsilon=1e-8
):
    """Fit one fixed-reference SE(2)+z trajectory per positive actor ID.

    Args:
        world_xyz: Persistent Gaussian trajectories with shape ``[T, N, 3]``.
        actor_ids: Positive fixed actor IDs with shape ``[N]``.
        weights: Optional visibility/opacity weights with shape ``[T, N]``.
    """
    if world_xyz.ndim != 3 or world_xyz.shape[-1] != 3:
        raise ValueError("world_xyz must have shape [T, N, 3]")
    if actor_ids.ndim != 1 or actor_ids.shape[0] != world_xyz.shape[1]:
        raise ValueError("actor_ids must have shape [N]")
    if actor_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("actor_ids must use an integer dtype")
    if torch.any(actor_ids <= 0):
        raise ValueError("actor_ids must be positive")
    if not bool(torch.isfinite(world_xyz).all()):
        raise ValueError("world_xyz must be finite")
    if not torch.is_floating_point(world_xyz):
        raise ValueError("world_xyz must use a floating-point dtype")
    if min_points < 2:
        raise ValueError("min_points must be at least two")

    if weights is None:
        weights = torch.ones(
            world_xyz.shape[:2], dtype=world_xyz.dtype, device=world_xyz.device
        )
    if weights.shape != world_xyz.shape[:2]:
        raise ValueError("weights must have shape [T, N]")
    if torch.any(weights < 0) or not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must be finite and non-negative")

    unique_ids = torch.unique(actor_ids, sorted=True)
    time_count = world_xyz.shape[0]
    actor_count = unique_ids.shape[0]
    canonical_xyz = torch.full_like(world_xyz[0], float("nan"))
    translations = torch.full(
        (time_count, actor_count, 3),
        float("nan"),
        dtype=world_xyz.dtype,
        device=world_xyz.device,
    )
    yaw = torch.full(
        (time_count, actor_count),
        float("nan"),
        dtype=world_xyz.dtype,
        device=world_xyz.device,
    )
    valid = torch.zeros(
        (time_count, actor_count), dtype=torch.bool, device=world_xyz.device
    )
    rmse_xy = torch.full_like(yaw, float("nan"))
    rmse_z = torch.full_like(yaw, float("nan"))
    rmse_3d = torch.full_like(yaw, float("nan"))
    reference_indices = torch.empty(
        actor_count, dtype=torch.long, device=world_xyz.device
    )

    for actor_index, actor_id in enumerate(unique_ids):
        actor_mask = actor_ids == actor_id
        actor_weights = weights[:, actor_mask]
        support_count = torch.sum(actor_weights > 0, dim=1)
        eligible = support_count >= min_points
        if not bool(eligible.any()):
            raise ValueError("actor {} never has sufficient support".format(int(actor_id)))
        support_weight = actor_weights.sum(dim=1)
        reference_index = torch.argmax(
            torch.where(eligible, support_weight, torch.full_like(support_weight, -1.0))
        )
        reference_indices[actor_index] = reference_index
        reference = world_xyz[reference_index, actor_mask]
        canonical_xyz[actor_mask] = reference
        reference_weights = actor_weights[reference_index]

        for time_index in range(time_count):
            joint_weights = torch.sqrt(reference_weights * actor_weights[time_index])
            if int(torch.sum(joint_weights > 0)) < min_points:
                continue
            try:
                translation, fitted_yaw, predicted = fit_planar_rigid(
                    reference,
                    world_xyz[time_index, actor_mask],
                    joint_weights,
                    epsilon=epsilon,
                )
            except ValueError:
                continue
            residual = predicted - world_xyz[time_index, actor_mask]
            normalized_weights = joint_weights / joint_weights.sum()
            translations[time_index, actor_index] = translation
            yaw[time_index, actor_index] = fitted_yaw
            valid[time_index, actor_index] = True
            rmse_xy[time_index, actor_index] = torch.sqrt(
                torch.sum(normalized_weights * residual[:, :2].square().sum(dim=1))
            )
            rmse_z[time_index, actor_index] = torch.sqrt(
                torch.sum(normalized_weights * residual[:, 2].square())
            )
            rmse_3d[time_index, actor_index] = torch.sqrt(
                torch.sum(normalized_weights * residual.square().sum(dim=1))
            )

    if bool(torch.isnan(canonical_xyz).any()):
        raise RuntimeError("not all actor points received a canonical reference")
    return {
        "actor_ids": unique_ids,
        "canonical_xyz": canonical_xyz,
        "reference_indices": reference_indices,
        "translations": translations,
        "yaw": yaw,
        "valid": valid,
        "rmse_xy": rmse_xy,
        "rmse_z": rmse_z,
        "rmse_3d": rmse_3d,
    }


def summarize_fit(fit):
    def summarize_metric(values):
        if values.numel() == 0:
            return {"median": None, "max": None}
        return {
            "median": float(torch.median(values)),
            "max": float(torch.max(values)),
        }

    actors = []
    for actor_index, actor_id in enumerate(fit["actor_ids"]):
        mask = fit["valid"][:, actor_index]
        valid_frames = int(mask.sum())
        actors.append(
            {
                "actor_id": int(actor_id),
                "reference_index": int(fit["reference_indices"][actor_index]),
                "valid_frames": valid_frames,
                "total_frames": int(mask.shape[0]),
                "valid_fraction": valid_frames / mask.shape[0],
                "rmse_xy": summarize_metric(fit["rmse_xy"][:, actor_index][mask]),
                "rmse_z": summarize_metric(fit["rmse_z"][:, actor_index][mask]),
                "rmse_3d": summarize_metric(fit["rmse_3d"][:, actor_index][mask]),
            }
        )
    return actors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_npz", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--min-points", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.trajectory_npz)
    world_xyz = torch.from_numpy(data["world_xyz"])
    actor_ids = torch.from_numpy(data["actor_ids"]).long()
    weights = (
        torch.from_numpy(data["weights"]).to(dtype=world_xyz.dtype)
        if "weights" in data
        else None
    )
    fit = fit_actor_rigid_trajectories(
        world_xyz, actor_ids, weights=weights, min_points=args.min_points
    )
    payload = {
        "trajectory_npz": str(args.trajectory_npz),
        "shape": list(world_xyz.shape),
        "actors": summarize_fit(fit),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.details_output,
            **{key: value.detach().cpu().numpy() for key, value in fit.items()}
        )


if __name__ == "__main__":
    main()
