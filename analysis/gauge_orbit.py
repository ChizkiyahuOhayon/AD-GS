#!/usr/bin/env python3
"""Numerically audit the proposed per-actor scale--range gauge.

This is a Stage-A falsification test.  It distinguishes the exact pinhole
footprint invariance from the extra assumptions needed for a complete rendered
image to remain unchanged.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils.graphics_utils import getProjectionMatrix


def project_moments(means, covariances, camera_center, fx, fy):
    camera_means = means - camera_center
    x, y, z = camera_means.unbind(dim=-1)
    if not bool(torch.all(z > 0)):
        raise ValueError("All test Gaussians must have positive camera depth.")

    projected_means = torch.stack((fx * x / z, fy * y / z), dim=-1)
    jacobian = torch.zeros(
        (means.shape[0], 2, 3), dtype=means.dtype, device=means.device
    )
    jacobian[:, 0, 0] = fx / z
    jacobian[:, 0, 2] = -fx * x / z.square()
    jacobian[:, 1, 1] = fy / z
    jacobian[:, 1, 2] = -fy * y / z.square()
    projected_covariances = jacobian @ covariances @ jacobian.transpose(1, 2)
    return projected_means, projected_covariances


def transform_actor(means, covariances, camera_center, scale):
    transformed_means = camera_center + scale * (means - camera_center)
    transformed_covariances = scale**2 * covariances
    return transformed_means, transformed_covariances


def render(means, scales, colors, opacities, image_size=96):
    device = means.device
    fov = math.pi / 3.0
    view = torch.eye(4, dtype=torch.float32, device=device)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).transpose(0, 1)
    projection = projection.to(device=device, dtype=torch.float32)
    settings = GaussianRasterizationSettings(
        image_height=image_size,
        image_width=image_size,
        tanfovx=math.tan(fov / 2.0),
        tanfovy=math.tan(fov / 2.0),
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=view @ projection,
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device=device),
        prefiltered=False,
        inv_depth=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(settings)
    rotations = torch.zeros((means.shape[0], 4), device=device)
    rotations[:, 0] = 1.0
    means2d = torch.zeros_like(means)
    image, _, depth, alpha, _, _ = rasterizer(
        means3D=means,
        means2D=means2d,
        opacities=opacities,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
    )
    torch.cuda.synchronize()
    return image, depth, alpha


def max_abs(left, right):
    return float((left - right).abs().max().item())


def run(lambdas, baseline):
    analytical_dtype = torch.float64
    actor_means = torch.tensor(
        [[-0.35, -0.12, 3.0], [0.28, -0.08, 4.1], [0.05, 0.30, 5.2]],
        dtype=analytical_dtype,
    )
    actor_scales = torch.tensor(
        [[0.14, 0.10, 0.12], [0.18, 0.12, 0.15], [0.20, 0.16, 0.13]],
        dtype=analytical_dtype,
    )
    actor_covariances = torch.diag_embed(actor_scales.square())
    primary_camera = torch.zeros(3, dtype=analytical_dtype)
    second_camera = torch.tensor([baseline, 0.0, 0.0], dtype=analytical_dtype)
    fx = fy = 1200.0

    base_mean_2d, base_cov_2d = project_moments(
        actor_means, actor_covariances, primary_camera, fx, fy
    )
    second_base_mean_2d, second_base_cov_2d = project_moments(
        actor_means, actor_covariances, second_camera, fx, fy
    )

    analytical = []
    for scale in lambdas:
        moved_means, moved_covariances = transform_actor(
            actor_means, actor_covariances, primary_camera, scale
        )
        mean_2d, cov_2d = project_moments(
            moved_means, moved_covariances, primary_camera, fx, fy
        )
        second_mean_2d, second_cov_2d = project_moments(
            moved_means, moved_covariances, second_camera, fx, fy
        )
        analytical.append(
            {
                "lambda": scale,
                "primary_mean_max_abs_px": max_abs(mean_2d, base_mean_2d),
                "primary_covariance_max_abs_px2": max_abs(cov_2d, base_cov_2d),
                "second_camera_mean_max_abs_px": max_abs(
                    second_mean_2d, second_base_mean_2d
                ),
                "second_camera_covariance_max_abs_px2": max_abs(
                    second_cov_2d, second_base_cov_2d
                ),
            }
        )

    if not torch.cuda.is_available():
        raise RuntimeError("The host-rasterizer audit requires CUDA.")
    device = torch.device("cuda")
    means_cuda = actor_means.float().to(device)
    scales_cuda = actor_scales.float().to(device)
    colors = torch.tensor(
        [[0.9, 0.2, 0.1], [0.1, 0.8, 0.2], [0.2, 0.3, 0.9]],
        dtype=torch.float32,
        device=device,
    )
    opacities = torch.full((3, 1), 0.85, dtype=torch.float32, device=device)
    base_image, base_depth, _ = render(
        means_cuda, scales_cuda, colors, opacities
    )

    rendered = []
    for scale in lambdas:
        image, depth, _ = render(
            scale * means_cuda, scale * scales_cuda, colors, opacities
        )
        rendered.append(
            {
                "lambda": scale,
                "actor_only_image_max_abs": max_abs(image, base_image),
                "actor_only_depth_max_abs": max_abs(depth, base_depth),
            }
        )

    actor_mean = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    actor_scale = torch.full((1, 3), 0.22, device=device)
    static_mean = torch.tensor([[0.0, 0.0, 4.0]], device=device)
    static_scale = torch.full((1, 3), 0.22 * 4.0 / 3.0, device=device)
    overlap_colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device
    )
    overlap_opacities = torch.full((2, 1), 0.95, device=device)
    ordered_image, _, _ = render(
        torch.cat((actor_mean, static_mean)),
        torch.cat((actor_scale, static_scale)),
        overlap_colors,
        overlap_opacities,
    )
    crossing_scale = 2.0
    crossed_image, _, _ = render(
        torch.cat((crossing_scale * actor_mean, static_mean)),
        torch.cat((crossing_scale * actor_scale, static_scale)),
        overlap_colors,
        overlap_opacities,
    )
    ordering_counterexample = {
        "lambda": crossing_scale,
        "image_max_abs_after_static_depth_crossing": max_abs(
            crossed_image, ordered_image
        ),
    }

    primary_mean_error = max(
        item["primary_mean_max_abs_px"] for item in analytical
    )
    primary_covariance_error = max(
        item["primary_covariance_max_abs_px2"] for item in analytical
    )
    actor_image_error = max(
        item["actor_only_image_max_abs"] for item in rendered
    )
    gate = {
        "projected_moments_near_machine_precision": (
            primary_mean_error <= 1e-10 and primary_covariance_error <= 1e-10
        ),
        "conditional_host_image_invariance": actor_image_error <= 1e-5,
        "unconditional_complete_image_invariance": False,
    }
    return {
        "orbit": "mu'=o+lambda*(mu-o), Sigma'=lambda^2*Sigma",
        "analytical_dtype": str(analytical_dtype),
        "second_camera_baseline_m": baseline,
        "analytical_projection": analytical,
        "adgs_rasterizer": rendered,
        "ordering_counterexample": ordering_counterexample,
        "assumptions_for_complete_image_invariance": [
            "positive lambda and positive camera depth",
            "one camera center per observation time, or negligible relative baseline",
            "unchanged clipping and visibility",
            "unchanged ordering against static and other independently transformed content",
            "appearance depends only on the preserved viewing direction",
        ],
        "gate": gate,
        "conclusion": (
            "The projected Gaussian orbit is exact. Complete-image invariance is "
            "conditional, and the multi-camera case is only approximate as baseline "
            "grows relative to actor range."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.75, 1.25, 2.0])
    parser.add_argument("--baseline", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run(args.lambdas, args.baseline)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not all(
        (
            result["gate"]["projected_moments_near_machine_precision"],
            result["gate"]["conditional_host_image_invariance"],
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
