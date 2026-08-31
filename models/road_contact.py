"""Fixed-actor road contact used by the GF-DGS Stage-B model."""

import torch

from models.contact_tie import _fixed_weighted_quantile, vertical_standard_deviation


def project_actor_contact_to_chart(
    xyz,
    scales,
    rotations,
    actor_ids,
    active_actor_ids,
    sample_weights,
    road_chart,
    quantile=0.05,
    sigma_multiplier=2.0,
):
    """Derive actor z translations from a metric road chart."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if scales.shape != xyz.shape or rotations.shape != (xyz.shape[0], 4):
        raise ValueError("scales and rotations must have shape [N, 3] and [N, 4]")
    if actor_ids.shape != xyz.shape[:1] or sample_weights.shape != xyz.shape[:1]:
        raise ValueError("actor_ids and sample_weights must have shape [N]")
    if torch.any(actor_ids <= 0) or torch.any(sample_weights < 0):
        raise ValueError("actor IDs must be positive and weights non-negative")

    support = xyz[:, 2] - sigma_multiplier * vertical_standard_deviation(
        scales, rotations
    )
    offset_z = torch.zeros_like(xyz[:, 2])
    before = []
    invalid_count = 0
    for actor_id in active_actor_ids:
        actor_mask = actor_ids == actor_id
        if not bool(actor_mask.any()):
            continue
        weights = sample_weights[actor_mask].detach()
        total_weight = weights.sum()
        if not bool(torch.isfinite(total_weight)) or float(total_weight) <= torch.finfo(
            weights.dtype
        ).eps:
            continue
        weights = weights / total_weight
        center_xy = torch.sum(xyz[actor_mask, :2] * weights[:, None], dim=0)
        road_height, valid = road_chart(center_xy[None])
        if not bool(valid.item()):
            invalid_count += 1
            continue

        lower_extent = _fixed_weighted_quantile(
            support[actor_mask], weights, quantile
        )
        shift = road_height[0] - lower_extent
        offset_z = offset_z + actor_mask.to(xyz.dtype) * shift
        before.append(torch.abs(shift))

    projected = xyz + torch.stack(
        (torch.zeros_like(offset_z), torch.zeros_like(offset_z), offset_z), dim=1
    )
    zero = xyz.new_zeros(())
    diagnostics = {
        "actor_count": len(before),
        "invalid_actor_count": invalid_count,
        "mean_abs_before": torch.stack(before).mean() if before else zero,
        "mean_abs_after": zero,
    }
    return projected, diagnostics
