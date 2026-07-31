"""Matched sparse-depth metrics for the GF-DGS kill test."""

import numpy as np


def select_sparse_actor_depth(uv, depth, actor_map, stable_actor_ids):
    """Z-buffer projected LiDAR samples and retain fixed actor-mask pixels."""
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    actor_map = np.asarray(actor_map)
    if uv.ndim != 2 or uv.shape[1] != 2 or depth.shape != uv.shape[:1]:
        raise ValueError("uv and depth must have shapes [N, 2] and [N]")
    if actor_map.ndim != 2:
        raise ValueError("actor_map must have shape [H, W]")

    pixels = np.rint(uv).astype(np.int64)
    height, width = actor_map.shape
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels = pixels[valid]
    depth = depth[valid]
    if depth.size == 0:
        return _empty_samples()

    flat = pixels[:, 1] * width + pixels[:, 0]
    order = np.lexsort((depth, flat))
    flat = flat[order]
    first = np.concatenate(([True], flat[1:] != flat[:-1]))
    pixels = pixels[order][first]
    depth = depth[order][first]
    actor_ids = actor_map[pixels[:, 1], pixels[:, 0]].astype(np.int64)

    keep = np.isin(actor_ids, np.asarray(stable_actor_ids, dtype=np.int64))
    return {
        "u": pixels[keep, 0],
        "v": pixels[keep, 1],
        "depth": depth[keep],
        "actor_id": actor_ids[keep],
    }


def _empty_samples():
    return {
        "u": np.empty(0, dtype=np.int64),
        "v": np.empty(0, dtype=np.int64),
        "depth": np.empty(0, dtype=np.float64),
        "actor_id": np.empty(0, dtype=np.int64),
    }


def alpha_composited_depth(inverse_depth, opacity):
    """Convert alpha-composited inverse depth to harmonic-mean depth."""
    inverse_depth = np.asarray(inverse_depth, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    if inverse_depth.shape != opacity.shape:
        raise ValueError("inverse_depth and opacity must have matching shapes")
    result = np.full(inverse_depth.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(inverse_depth)
        & np.isfinite(opacity)
        & (inverse_depth > 0.0)
        & (opacity > 0.0)
    )
    result[valid] = opacity[valid] / inverse_depth[valid]
    return result


def paired_sparse_depth_metrics(
    ground_truth,
    actor_ids,
    baseline_inverse_depth,
    baseline_opacity,
    oracle_inverse_depth,
    oracle_opacity,
    min_opacity=0.1,
    min_pixels=100,
    min_actors=5,
):
    """Compute mean AbsRel on one paired validity mask for both arms."""
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    actor_ids = np.asarray(actor_ids, dtype=np.int64)
    arrays = (
        actor_ids,
        np.asarray(baseline_inverse_depth),
        np.asarray(baseline_opacity),
        np.asarray(oracle_inverse_depth),
        np.asarray(oracle_opacity),
    )
    if ground_truth.ndim != 1 or any(array.shape != ground_truth.shape for array in arrays):
        raise ValueError("all inputs must have matching shape [N]")

    baseline_depth = alpha_composited_depth(baseline_inverse_depth, baseline_opacity)
    oracle_depth = alpha_composited_depth(oracle_inverse_depth, oracle_opacity)
    valid = (
        np.isfinite(ground_truth)
        & (ground_truth > 0.0)
        & np.isfinite(baseline_depth)
        & np.isfinite(oracle_depth)
        & (np.asarray(baseline_opacity) >= min_opacity)
        & (np.asarray(oracle_opacity) >= min_opacity)
    )
    pixel_count = int(valid.sum())
    if pixel_count < min_pixels:
        raise ValueError("insufficient common pixels: {} < {}".format(pixel_count, min_pixels))
    valid_actor_ids = sorted(int(value) for value in np.unique(actor_ids[valid]))
    if len(valid_actor_ids) < min_actors:
        raise ValueError(
            "insufficient common actors: {} < {}".format(len(valid_actor_ids), min_actors)
        )

    gt = ground_truth[valid]
    baseline_absrel = float(np.mean(np.abs(baseline_depth[valid] - gt) / gt))
    oracle_absrel = float(np.mean(np.abs(oracle_depth[valid] - gt) / gt))
    relative_improvement = (baseline_absrel - oracle_absrel) / baseline_absrel
    return {
        "pixel_count": pixel_count,
        "actor_count": len(valid_actor_ids),
        "actor_ids": valid_actor_ids,
        "baseline_absrel": baseline_absrel,
        "oracle_absrel": oracle_absrel,
        "relative_improvement": float(relative_improvement),
    }
