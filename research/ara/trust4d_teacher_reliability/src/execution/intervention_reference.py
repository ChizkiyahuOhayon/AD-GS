"""Minimal reference math for the preregistered Trust4D EXP-002 diagnostic."""

from typing import NamedTuple

import torch


class SimilarityFit(NamedTuple):
    rotation: torch.Tensor
    scale: torch.Tensor
    translation: torch.Tensor
    rotation_residual_deg: torch.Tensor
    center_rmse: torch.Tensor


class InterventionStats(NamedTuple):
    disagreement: torch.Tensor
    covariance: torch.Tensor
    principal_direction: torch.Tensor
    eigenvalues: torch.Tensor
    directional_capture: torch.Tensor


def _camera_centers(extrinsics: torch.Tensor) -> torch.Tensor:
    """Return world-space centers from OpenCV camera-from-world matrices."""
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    return -torch.einsum("nij,nj->ni", rotations.transpose(1, 2), translations)


def project_to_so3(matrices: torch.Tensor) -> torch.Tensor:
    """Project the arithmetic mean of 3x3 matrices to the nearest rotation."""
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or matrices.shape[0] == 0:
        raise ValueError("matrices must have shape [N, 3, 3] with N > 0")
    mean_matrix = matrices.mean(dim=0)
    u, _, vh = torch.linalg.svd(mean_matrix)
    correction = torch.eye(3, dtype=matrices.dtype, device=matrices.device)
    correction[-1, -1] = torch.det(u @ vh)
    return u @ correction @ vh


def fit_camera_guided_similarity(
    predicted_extrinsics: torch.Tensor,
    target_extrinsics: torch.Tensor,
    eps: float = 1e-12,
) -> SimilarityFit:
    """Fit X_target = scale * rotation @ X_predicted + translation.

    Both inputs are OpenCV camera-from-world matrices with shape [N, 3, 4]
    or [N, 4, 4]. Rotation is fixed first from camera orientations; positive
    scale and translation are then fitted from camera centers.
    """
    if predicted_extrinsics.ndim != 3 or target_extrinsics.ndim != 3:
        raise ValueError("extrinsics must have shape [N, 3|4, 4]")
    if predicted_extrinsics.shape[0] < 2 or predicted_extrinsics.shape != target_extrinsics.shape:
        raise ValueError("predicted and target extrinsics must share shape and contain >=2 views")
    if predicted_extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("extrinsics must have shape [N, 3, 4] or [N, 4, 4]")
    if not torch.isfinite(predicted_extrinsics).all() or not torch.isfinite(target_extrinsics).all():
        raise ValueError("extrinsics must be finite")

    predicted_rotations = predicted_extrinsics[:, :3, :3]
    target_rotations = target_extrinsics[:, :3, :3]
    rotation_candidates = target_rotations.transpose(1, 2) @ predicted_rotations
    rotation = project_to_so3(rotation_candidates)

    predicted_centers = _camera_centers(predicted_extrinsics)
    target_centers = _camera_centers(target_extrinsics)
    rotated_centers = predicted_centers @ rotation.T
    predicted_centered = rotated_centers - rotated_centers.mean(dim=0)
    target_centered = target_centers - target_centers.mean(dim=0)
    denominator = predicted_centered.square().sum()
    if denominator <= eps:
        raise ValueError("predicted camera centers do not constrain scale")
    scale = (predicted_centered * target_centered).sum() / denominator
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("fitted scale must be positive and finite")
    translation = target_centers.mean(dim=0) - scale * (
        rotation @ predicted_centers.mean(dim=0)
    )

    aligned_centers = scale * rotated_centers + translation
    center_rmse = torch.sqrt((aligned_centers - target_centers).square().sum(dim=1).mean())
    residual_rotations = rotation_candidates @ rotation.T
    traces = residual_rotations.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angles = ((traces - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation_residual_deg = torch.rad2deg(torch.acos(cos_angles)).median()
    return SimilarityFit(rotation, scale, translation, rotation_residual_deg, center_rmse)


def apply_similarity(points: torch.Tensor, fit: SimilarityFit) -> torch.Tensor:
    """Apply a fitted similarity transform to points with final dimension 3."""
    if points.shape[-1] != 3 or not torch.isfinite(points).all():
        raise ValueError("points must be finite with final dimension 3")
    return fit.scale * (points @ fit.rotation.T) + fit.translation


def intervention_statistics(
    displacements: torch.Tensor,
    original_error_vector: torch.Tensor,
    eps: float = 1e-12,
) -> InterventionStats:
    """Compute scalar spread, covariance direction, and error capture."""
    if displacements.ndim != 2 or displacements.shape[1] != 3 or displacements.shape[0] < 2:
        raise ValueError("displacements must have shape [K, 3] with K >= 2")
    if original_error_vector.shape != (3,):
        raise ValueError("original_error_vector must have shape [3]")
    if not torch.isfinite(displacements).all() or not torch.isfinite(original_error_vector).all():
        raise ValueError("inputs must be finite")

    disagreement = torch.pdist(displacements).median()
    centered = displacements - displacements.mean(dim=0)
    covariance = centered.T @ centered / (displacements.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    principal_direction = eigenvectors[:, -1]
    numerator = torch.dot(principal_direction, original_error_vector).square()
    directional_capture = numerator / (original_error_vector.square().sum() + eps)
    return InterventionStats(
        disagreement,
        covariance,
        principal_direction,
        eigenvalues,
        directional_capture,
    )
