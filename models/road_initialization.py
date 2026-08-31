"""Geometry-only initialization for the GF-DGS road chart."""

import torch

from models.road_chart import BicubicRoadChart


def actor_center_samples(xyz, times, actor_ids, min_points=5):
    """Return one horizontal median center for each supported actor-frame."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if times.shape != xyz.shape[:1] or actor_ids.shape != xyz.shape[:1]:
        raise ValueError("times and actor_ids must have shape [N]")
    if min_points < 1:
        raise ValueError("min_points must be positive")
    if torch.any(actor_ids <= 0):
        raise ValueError("actor_ids must be positive")

    centers = []
    for actor_id in torch.unique(actor_ids, sorted=True):
        actor_mask = actor_ids == actor_id
        for time in torch.unique(times[actor_mask], sorted=True):
            frame_xyz = xyz[actor_mask & (times == time)]
            if frame_xyz.shape[0] >= min_points:
                centers.append(torch.median(frame_xyz[:, :2], dim=0).values)
    if not centers:
        raise ValueError("no actor-frame has enough points")
    return torch.stack(centers)


def _lower_envelope_cells(static_xyz, cell_size, min_points, quantile):
    cell_indices = torch.floor(static_xyz[:, :2] / cell_size).long()
    minimum = torch.amin(cell_indices, dim=0)
    normalized = cell_indices - minimum
    stride = torch.amax(normalized[:, 1]) + 1
    keys = normalized[:, 0] * stride + normalized[:, 1]

    z_order = torch.argsort(static_xyz[:, 2])
    key_order = torch.argsort(keys[z_order], stable=True)
    order = z_order[key_order]
    sorted_keys = keys[order]
    _, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    keep = counts >= min_points
    counts = counts[keep]
    starts = starts[keep]
    if starts.numel() == 0:
        raise ValueError("no static cell has enough points")

    rank = torch.floor((counts - 1) * quantile).long()
    selected = order[starts + rank]
    selected_cells = cell_indices[selected]
    cell_xy = (selected_cells.to(static_xyz.dtype) + 0.5) * cell_size
    return cell_xy, static_xyz[selected, 2]


def extract_road_support(
    static_xyz,
    query_xy,
    cell_size=1.0,
    search_radius=3.0,
    min_points_per_cell=3,
    min_neighbor_cells=3,
    cell_quantile=0.1,
    neighbor_count=9,
    chunk_size=256,
):
    """Estimate road height near actor locations from static LiDAR geometry.

    Static points are reduced to a robust lower-envelope sample per horizontal
    cell.  The road estimate at each query is the median of nearby envelope
    samples.  No semantic road label is consumed.
    """
    if static_xyz.ndim != 2 or static_xyz.shape[1] != 3:
        raise ValueError("static_xyz must have shape [N, 3]")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must have shape [M, 2]")
    if static_xyz.device != query_xy.device or static_xyz.dtype != query_xy.dtype:
        raise ValueError("static_xyz and query_xy must share device and dtype")
    if not torch.is_floating_point(static_xyz):
        raise ValueError("inputs must use a floating-point dtype")
    if not bool(torch.isfinite(static_xyz).all()) or not bool(torch.isfinite(query_xy).all()):
        raise ValueError("inputs must be finite")
    if cell_size <= 0.0 or search_radius <= 0.0:
        raise ValueError("cell_size and search_radius must be positive")
    if min_points_per_cell < 1 or min_neighbor_cells < 1:
        raise ValueError("minimum support must be positive")
    if not 0.0 <= cell_quantile <= 1.0:
        raise ValueError("cell_quantile must be in [0, 1]")
    if neighbor_count < min_neighbor_cells:
        raise ValueError("neighbor_count must cover min_neighbor_cells")

    cell_xy, cell_z = _lower_envelope_cells(
        static_xyz, cell_size, min_points_per_cell, cell_quantile
    )
    neighbor_count = min(neighbor_count, cell_xy.shape[0])
    support_xy = []
    support_z = []
    for start in range(0, query_xy.shape[0], chunk_size):
        query = query_xy[start : start + chunk_size]
        distances = torch.cdist(query, cell_xy)
        nearest_distance, nearest_index = torch.topk(
            distances, k=neighbor_count, dim=1, largest=False
        )
        nearby = nearest_distance <= search_radius
        counts = nearby.sum(dim=1)
        heights = cell_z[nearest_index]
        heights = torch.where(nearby, heights, torch.full_like(heights, torch.inf))
        heights = torch.sort(heights, dim=1).values
        valid = counts >= min_neighbor_cells
        if bool(valid.any()):
            median_index = torch.div(counts[valid] - 1, 2, rounding_mode="floor")
            valid_heights = heights[valid]
            support_xy.append(query[valid])
            support_z.append(valid_heights.gather(1, median_index[:, None])[:, 0])

    if not support_xy:
        raise ValueError("no actor query has local static road support")
    return torch.cat(support_xy), torch.cat(support_z)


def initialize_road_chart(support_xy, support_z, knot_spacing=2.0, neighbors=8):
    """Initialize a road chart with a planar trend and local residuals."""
    if support_xy.ndim != 2 or support_xy.shape[1] != 2:
        raise ValueError("support_xy must have shape [N, 2]")
    if support_z.shape != support_xy.shape[:1]:
        raise ValueError("support_z must have shape [N]")
    if support_xy.shape[0] < 3:
        raise ValueError("at least three road samples are required")
    if support_xy.device != support_z.device or support_xy.dtype != support_z.dtype:
        raise ValueError("support_xy and support_z must share device and dtype")
    if knot_spacing <= 0.0 or neighbors < 1:
        raise ValueError("knot_spacing and neighbors must be positive")

    lower = torch.floor(torch.amin(support_xy, dim=0) / knot_spacing) * knot_spacing
    origin = lower - knot_spacing
    upper = (
        torch.ceil(torch.amax(support_xy, dim=0) / knot_spacing) * knot_spacing
        + 2.0 * knot_spacing
    )
    shape_xy = torch.round((upper - origin) / knot_spacing).long() + 1
    grid_y, grid_x = torch.meshgrid(
        torch.arange(shape_xy[1], device=support_xy.device, dtype=support_xy.dtype),
        torch.arange(shape_xy[0], device=support_xy.device, dtype=support_xy.dtype),
        indexing="ij",
    )
    control_xy = torch.stack((grid_x, grid_y), dim=-1) * knot_spacing + origin
    flat_control_xy = control_xy.reshape(-1, 2)

    design = torch.cat((support_xy, torch.ones_like(support_z[:, None])), dim=1)
    plane = torch.linalg.lstsq(design, support_z[:, None]).solution[:, 0]
    residual = support_z - design @ plane
    neighbor_count = min(neighbors, support_xy.shape[0])
    distance, index = torch.topk(
        torch.cdist(flat_control_xy, support_xy),
        k=neighbor_count,
        dim=1,
        largest=False,
    )
    weight = torch.reciprocal(distance.square() + 1e-6)
    local_residual = torch.sum(weight * residual[index], dim=1) / torch.sum(weight, dim=1)
    control_design = torch.cat(
        (flat_control_xy, torch.ones_like(flat_control_xy[:, :1])), dim=1
    )
    control_heights = (control_design @ plane + local_residual).reshape(
        int(shape_xy[1]), int(shape_xy[0])
    )
    return BicubicRoadChart(control_heights, origin, knot_spacing)
