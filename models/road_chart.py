import torch
from torch import nn


def _cubic_bspline_weights(fraction):
    one_minus = 1.0 - fraction
    return torch.stack(
        (
            one_minus ** 3,
            3.0 * fraction ** 3 - 6.0 * fraction ** 2 + 4.0,
            -3.0 * fraction ** 3 + 3.0 * fraction ** 2 + 3.0 * fraction + 1.0,
            fraction ** 3,
        ),
        dim=-1,
    ) / 6.0


class BicubicRoadChart(nn.Module):
    """Uniform tensor-product cubic B-spline road-height chart."""

    def __init__(self, control_heights, origin_xy, knot_spacing=2.0):
        super().__init__()
        control_heights = torch.as_tensor(control_heights)
        if not control_heights.is_floating_point():
            control_heights = control_heights.float()
        origin_xy = torch.as_tensor(
            origin_xy,
            dtype=control_heights.dtype,
            device=control_heights.device,
        )
        if control_heights.ndim != 2 or min(control_heights.shape) < 4:
            raise ValueError("control_heights must be a [H, W] grid with H,W >= 4")
        if origin_xy.shape != (2,):
            raise ValueError("origin_xy must have shape [2]")
        if knot_spacing <= 0.0:
            raise ValueError("knot_spacing must be positive")

        self.control_heights = nn.Parameter(control_heights.clone())
        self.register_buffer("origin_xy", origin_xy.clone())
        self.register_buffer(
            "knot_spacing",
            control_heights.new_tensor(float(knot_spacing)),
        )

    def forward(self, xy):
        if xy.shape[-1] != 2:
            raise ValueError("xy must have shape [..., 2]")

        xy = xy.to(dtype=self.control_heights.dtype, device=self.control_heights.device)
        uv = (xy - self.origin_xy) / self.knot_spacing
        height, width = self.control_heights.shape
        valid = (
            (uv[..., 0] >= 1.0)
            & (uv[..., 0] < width - 2.0)
            & (uv[..., 1] >= 1.0)
            & (uv[..., 1] < height - 2.0)
        )

        safe_u = uv[..., 0].clamp(1.0, width - 2.0 - 1e-6)
        safe_v = uv[..., 1].clamp(1.0, height - 2.0 - 1e-6)
        cell_u = torch.floor(safe_u).long()
        cell_v = torch.floor(safe_v).long()
        weight_u = _cubic_bspline_weights(safe_u - cell_u)
        weight_v = _cubic_bspline_weights(safe_v - cell_v)

        offsets = torch.arange(-1, 3, device=xy.device)
        index_u = cell_u[..., None] + offsets
        index_v = cell_v[..., None] + offsets
        controls = self.control_heights[
            index_v[..., :, None],
            index_u[..., None, :],
        ]
        road_height = torch.sum(
            controls * weight_v[..., :, None] * weight_u[..., None, :],
            dim=(-2, -1),
        )
        return road_height, valid
