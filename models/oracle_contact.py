"""Fixed-ID oracle contact projection for the single-scene kill test."""

from typing import Dict

import torch

from models.contact_tie import _fixed_weighted_quantile, vertical_standard_deviation


def _validate_points(xyz, times, actor_ids):
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if times.shape != xyz.shape[:1]:
        raise ValueError("times must have shape [N]")
    if actor_ids.shape != xyz.shape[:1]:
        raise ValueError("actor_ids must have shape [N]")
    if actor_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("actor_ids must use an integer dtype")
    if torch.any(actor_ids <= 0):
        raise ValueError("actor_ids must be positive")
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(times).all()):
        raise ValueError("xyz and times must be finite")


def build_oracle_contact_tracks(
    xyz: torch.Tensor,
    times: torch.Tensor,
    actor_ids: torch.Tensor,
    min_points_per_frame: int = 20,
    min_frames: int = 5,
    max_centroid_jump: float = 10.0,
    quantile: float = 0.05,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Build smooth per-actor lower-height tracks from initialization LiDAR points."""
    _validate_points(xyz, times, actor_ids)
    if min_points_per_frame < 1 or min_frames < 1:
        raise ValueError("minimum support must be positive")
    if max_centroid_jump <= 0.0:
        raise ValueError("max_centroid_jump must be positive")

    all_times = torch.unique(times, sorted=True)
    if all_times.numel() < 2:
        raise ValueError("at least two global time samples are required")
    base_step = torch.min(all_times[1:] - all_times[:-1])
    tracks = {}
    for actor_id in torch.unique(actor_ids, sorted=True):
        actor_mask = actor_ids == actor_id
        actor_times = torch.unique(times[actor_mask], sorted=True)
        kept_times = []
        heights = []
        centers = []
        for time in actor_times:
            frame_mask = actor_mask & (times == time)
            frame_xyz = xyz[frame_mask]
            if frame_xyz.shape[0] < min_points_per_frame:
                continue
            weights = torch.ones_like(frame_xyz[:, 2])
            kept_times.append(time)
            heights.append(_fixed_weighted_quantile(frame_xyz[:, 2], weights, quantile))
            centers.append(torch.median(frame_xyz, dim=0).values)
        if len(kept_times) < min_frames:
            continue

        kept_times = torch.stack(kept_times)
        centers = torch.stack(centers)
        gaps = (kept_times[1:] - kept_times[:-1]) / base_step
        per_frame_jump = torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=1) / gaps
        if bool(torch.any(per_frame_jump > max_centroid_jump)):
            continue
        tracks[int(actor_id)] = {
            "times": kept_times.detach(),
            "heights": torch.stack(heights).detach(),
        }
    return tracks


def interpolate_track_height(query_time, times, heights):
    """Linearly interpolate a scalar track and clamp outside its time range."""
    if times.ndim != 1 or heights.shape != times.shape or times.numel() == 0:
        raise ValueError("times and heights must have matching non-empty shape [T]")
    if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("times must be strictly increasing")
    query = torch.as_tensor(query_time, dtype=times.dtype, device=times.device)
    if query <= times[0]:
        return heights[0]
    if query >= times[-1]:
        return heights[-1]
    right = torch.searchsorted(times, query, right=False)
    left = right - 1
    alpha = (query - times[left]) / (times[right] - times[left])
    return heights[left] + alpha * (heights[right] - heights[left])


def project_actor_contact(
    xyz: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    actor_ids: torch.Tensor,
    query_time,
    tracks: Dict[int, Dict[str, torch.Tensor]],
    quantile: float = 0.05,
    sigma_multiplier: float = 2.0,
):
    """Translate stable actors in z so their lower support matches oracle height."""
    _validate_points(xyz, torch.zeros_like(xyz[:, 0]), actor_ids)
    if scales.shape != (xyz.shape[0], 3):
        raise ValueError("scales must have shape [N, 3]")
    if rotations.shape != (xyz.shape[0], 4):
        raise ValueError("rotations must have shape [N, 4]")

    support = xyz[:, 2] - sigma_multiplier * vertical_standard_deviation(scales, rotations)
    offset_z = torch.zeros_like(xyz[:, 2])
    before = []
    after = []
    for actor_id, track in tracks.items():
        actor_mask = actor_ids == actor_id
        if not bool(actor_mask.any()):
            continue
        track_times = track["times"].to(device=xyz.device, dtype=xyz.dtype)
        track_heights = track["heights"].to(device=xyz.device, dtype=xyz.dtype)
        target = interpolate_track_height(query_time, track_times, track_heights)
        weights = torch.ones_like(support[actor_mask])
        lower = _fixed_weighted_quantile(support[actor_mask], weights, quantile)
        shift = target - lower
        offset_z = offset_z + actor_mask.to(dtype=xyz.dtype) * shift
        before.append(torch.abs(lower - target))
        after.append(torch.abs(lower + shift - target))

    projected = xyz + torch.stack(
        (torch.zeros_like(offset_z), torch.zeros_like(offset_z), offset_z), dim=1
    )
    zero = xyz.new_zeros(())
    diagnostics = {
        "actor_count": len(before),
        "mean_abs_before": torch.stack(before).mean() if before else zero,
        "mean_abs_after": torch.stack(after).mean() if after else zero,
    }
    return projected, diagnostics
