"""Fixed-assignment actor-centric transforms for the Stage-B prototype."""

from typing import Tuple

import torch


def fixed_memberships_from_actor_ids(
    actor_ids: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map positive instance IDs to sorted columns of a one-hot membership matrix."""
    if actor_ids.ndim != 1:
        raise ValueError("actor_ids must have shape [N]")
    if actor_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("actor_ids must use an integer dtype")
    if torch.any(actor_ids <= 0):
        raise ValueError("fixed Stage-B actor IDs must be positive")

    unique_ids, inverse = torch.unique(actor_ids, sorted=True, return_inverse=True)
    memberships = torch.nn.functional.one_hot(
        inverse, num_classes=unique_ids.shape[0]
    ).to(dtype=dtype)
    return unique_ids, memberships


def _validate_pose_inputs(
    points: torch.Tensor,
    translations: torch.Tensor,
    yaw: torch.Tensor,
) -> None:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if translations.shape != points.shape:
        raise ValueError("translations must have shape [N, 3]")
    if yaw.shape != points.shape[:1]:
        raise ValueError("yaw must have shape [N]")


def canonicalize_actor_points(
    world_xyz: torch.Tensor,
    translations: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Apply the inverse per-point actor pose ``Rz(yaw), translation``."""
    _validate_pose_inputs(world_xyz, translations, yaw)
    centered = world_xyz - translations
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    canonical_x = cosine * centered[:, 0] + sine * centered[:, 1]
    canonical_y = -sine * centered[:, 0] + cosine * centered[:, 1]
    return torch.stack((canonical_x, canonical_y, centered[:, 2]), dim=-1)


def world_points_from_actor_pose(
    canonical_xyz: torch.Tensor,
    translations: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Apply the per-point actor pose ``Rz(yaw), translation``."""
    _validate_pose_inputs(canonical_xyz, translations, yaw)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    world_x = cosine * canonical_xyz[:, 0] - sine * canonical_xyz[:, 1]
    world_y = sine * canonical_xyz[:, 0] + cosine * canonical_xyz[:, 1]
    rotated = torch.stack((world_x, world_y, canonical_xyz[:, 2]), dim=-1)
    return rotated + translations
