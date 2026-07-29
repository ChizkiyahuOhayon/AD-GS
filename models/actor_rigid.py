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


def heading_from_actor_centers(
    actor_centers: torch.Tensor,
    times: torch.Tensor,
    speed_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive unwrapped yaw from horizontal center tangents.

    Low-speed samples copy the most recent moving heading.  Leading low-speed
    samples, which have no history, copy the first future moving heading.
    ``actor_centers`` must be world-space centers, not pose translations tied
    to an arbitrary canonical origin.
    """
    if actor_centers.ndim != 3 or actor_centers.shape[-1] != 3:
        raise ValueError("actor_centers must have shape [T, K, 3]")
    if actor_centers.shape[0] < 2:
        raise ValueError("at least two time samples are required")
    if times.ndim != 1 or times.shape[0] != actor_centers.shape[0]:
        raise ValueError("times must have shape [T]")
    if not torch.is_floating_point(actor_centers) or not torch.is_floating_point(times):
        raise ValueError("actor_centers and times must use floating-point dtypes")
    if times.device != actor_centers.device:
        raise ValueError("actor_centers and times must be on the same device")
    if not bool(torch.isfinite(actor_centers).all()) or not bool(
        torch.isfinite(times).all()
    ):
        raise ValueError("actor_centers and times must be finite")
    if not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("times must be strictly increasing")
    if speed_threshold <= 0.0:
        raise ValueError("speed_threshold must be positive")

    velocity_xy = torch.empty_like(actor_centers[..., :2])
    segment_velocity = (actor_centers[1:, :, :2] - actor_centers[:-1, :, :2]) / (
        times[1:] - times[:-1]
    )[:, None, None]
    velocity_xy[0] = segment_velocity[0]
    velocity_xy[-1] = segment_velocity[-1]
    if actor_centers.shape[0] > 2:
        velocity_xy[1:-1] = (
            actor_centers[2:, :, :2] - actor_centers[:-2, :, :2]
        ) / (times[2:] - times[:-2])[:, None, None]

    speed = torch.sqrt(torch.sum(velocity_xy.square(), dim=-1))
    moving = speed >= speed_threshold
    raw_heading = torch.atan2(velocity_xy[..., 1], velocity_xy[..., 0])
    headings = []
    for actor_index in range(actor_centers.shape[1]):
        moving_indices = torch.nonzero(moving[:, actor_index], as_tuple=False).flatten()
        if moving_indices.numel() == 0:
            raise ValueError("actor {} has no moving heading sample".format(actor_index))
        first_moving = int(moving_indices[0])
        last_valid_heading = raw_heading[first_moving, actor_index]
        filled_values = []
        for time_index in range(actor_centers.shape[0]):
            if bool(moving[time_index, actor_index]):
                last_valid_heading = raw_heading[time_index, actor_index]
            filled_values.append(last_valid_heading)
        filled = torch.stack(filled_values)
        unwrapped = [filled[0]]
        for time_index in range(1, actor_centers.shape[0]):
            difference = filled[time_index] - unwrapped[-1]
            wrapped_difference = torch.atan2(
                torch.sin(difference), torch.cos(difference)
            )
            unwrapped.append(unwrapped[-1] + wrapped_difference)
        headings.append(torch.stack(unwrapped))
    return torch.stack(headings, dim=1), moving
