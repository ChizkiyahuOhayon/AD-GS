"""Stage-B road-contact height parameterizations.

This module deliberately assumes fixed actor memberships. The exact weighted
quantile used here is piecewise differentiable in the selected support value,
but it is not a valid differentiable assignment operator for Stage C.
"""

from typing import Optional, Tuple

import torch
from torch import nn


def vertical_standard_deviation(
    scales: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Return each Gaussian's standard deviation along world/canonical z.

    Args:
        scales: Activated positive Gaussian scales with shape ``[N, 3]``.
        rotations: Quaternions in AD-GS ``(w, x, y, z)`` order, shape ``[N, 4]``.
    """
    if scales.ndim != 2 or scales.shape[-1] != 3:
        raise ValueError("scales must have shape [N, 3]")
    if rotations.ndim != 2 or rotations.shape != (scales.shape[0], 4):
        raise ValueError("rotations must have shape [N, 4]")
    if torch.any(scales <= 0):
        raise ValueError("scales must be strictly positive activated scales")

    quaternions = torch.nn.functional.normalize(rotations, dim=-1)
    w, x, y, z = quaternions.unbind(dim=-1)

    # Third row of the rotation matrix used by utils.general_utils.build_rotation.
    row_z = torch.stack(
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    )
    variance_z = torch.sum((row_z * scales) ** 2, dim=-1)
    return torch.sqrt(torch.clamp_min(variance_z, torch.finfo(scales.dtype).tiny))


def _fixed_weighted_quantile(
    values: torch.Tensor,
    weights: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Exact weighted quantile for fixed Stage-B memberships."""
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("values and weights must have matching shape [N]")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")

    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    total_weight = sorted_weights.sum()
    if total_weight.detach().item() <= 0.0:
        raise ValueError("each actor must have positive total weight")

    threshold = total_weight * quantile
    cumulative_weight = torch.cumsum(sorted_weights, dim=0)
    index = torch.searchsorted(cumulative_weight, threshold, right=False)
    index = torch.clamp(index, max=sorted_values.shape[0] - 1)
    return sorted_values[index]


def canonical_lower_extent(
    canonical_xyz: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    memberships: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
    quantile: float = 0.05,
    sigma_multiplier: float = 2.0,
) -> torch.Tensor:
    """Compute the fixed-membership lower support ``b_c`` for each actor."""
    if canonical_xyz.ndim != 2 or canonical_xyz.shape[-1] != 3:
        raise ValueError("canonical_xyz must have shape [N, 3]")
    if memberships.ndim != 2 or memberships.shape[0] != canonical_xyz.shape[0]:
        raise ValueError("memberships must have shape [N, K]")
    if torch.any(memberships < 0):
        raise ValueError("memberships must be non-negative")
    if sigma_multiplier < 0.0:
        raise ValueError("sigma_multiplier must be non-negative")

    if sample_weights is None:
        sample_weights = torch.ones_like(canonical_xyz[:, 2])
    elif sample_weights.shape != canonical_xyz[:, 2].shape:
        raise ValueError("sample_weights must have shape [N]")
    if torch.any(sample_weights < 0):
        raise ValueError("sample_weights must be non-negative")

    support = canonical_xyz[:, 2] - sigma_multiplier * vertical_standard_deviation(
        scales, rotations
    )
    actor_extents = []
    for actor_index in range(memberships.shape[1]):
        actor_weights = memberships[:, actor_index] * sample_weights
        actor_extents.append(
            _fixed_weighted_quantile(support, actor_weights, quantile)
        )
    return torch.stack(actor_extents)


class ExtentTiedHeight(nn.Module):
    """Derive actor z from road height and canonical lower extent."""

    def __init__(self, quantile: float = 0.05, sigma_multiplier: float = 2.0):
        super().__init__()
        self.quantile = quantile
        self.sigma_multiplier = sigma_multiplier

    def forward(
        self,
        road_height: torch.Tensor,
        canonical_xyz: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        memberships: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_extent = canonical_lower_extent(
            canonical_xyz=canonical_xyz,
            scales=scales,
            rotations=rotations,
            memberships=memberships,
            sample_weights=sample_weights,
            quantile=self.quantile,
            sigma_multiplier=self.sigma_multiplier,
        )
        if road_height.shape != lower_extent.shape:
            raise ValueError("road_height must have shape [K]")
        return road_height - lower_extent, lower_extent


class FreeOffsetHeight(nn.Module):
    """Control with a freely learnable per-actor lower contact offset."""

    def __init__(self, initial_lower_extent: torch.Tensor):
        super().__init__()
        if initial_lower_extent.ndim != 1:
            raise ValueError("initial_lower_extent must have shape [K]")
        self.free_lower_extent = nn.Parameter(initial_lower_extent.detach().clone())

    def forward(self, road_height: torch.Tensor) -> torch.Tensor:
        if road_height.shape != self.free_lower_extent.shape:
            raise ValueError("road_height must have shape [K]")
        return road_height - self.free_lower_extent
