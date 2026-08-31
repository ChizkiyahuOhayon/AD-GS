#!/usr/bin/env python3
"""Inspect geometry-only GF-DGS road initialization on a real point cloud."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.road_initialization import (  # noqa: E402
    actor_center_samples,
    extract_road_support,
    initialize_road_chart,
)


def inspect(point_cloud_path, min_actor_points=5):
    vertices = PlyData.read(point_cloud_path)["vertex"]
    xyz = torch.from_numpy(
        np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1).copy()
    ).float()
    times = torch.from_numpy(np.asarray(vertices["t"]).copy()).float()
    actor_ids = torch.from_numpy(np.asarray(vertices["obj"]).copy()).long()
    finite = torch.isfinite(xyz).all(dim=1) & torch.isfinite(times)
    xyz = xyz[finite]
    times = times[finite]
    actor_ids = actor_ids[finite]

    static_xyz = xyz[actor_ids <= 0]
    dynamic = actor_ids > 0
    query_xy = actor_center_samples(
        xyz[dynamic], times[dynamic], actor_ids[dynamic], min_points=min_actor_points
    )
    support_xy, support_z = extract_road_support(static_xyz, query_xy)
    chart = initialize_road_chart(support_xy, support_z)
    fitted_z, valid = chart(support_xy)
    residual = torch.abs(fitted_z - support_z)

    return {
        "point_cloud": str(Path(point_cloud_path).resolve()),
        "static_points": int(static_xyz.shape[0]),
        "dynamic_points": int(dynamic.sum()),
        "actor_frame_queries": int(query_xy.shape[0]),
        "supported_queries": int(support_xy.shape[0]),
        "support_rate": float(support_xy.shape[0] / query_xy.shape[0]),
        "road_height": {
            "min": float(torch.amin(support_z)),
            "median": float(torch.median(support_z)),
            "max": float(torch.amax(support_z)),
        },
        "chart_shape": list(chart.control_heights.shape),
        "chart_valid_rate": float(valid.float().mean()),
        "fit_abs_residual": {
            "median": float(torch.median(residual)),
            "p90": float(torch.quantile(residual, 0.9)),
            "max": float(torch.amax(residual)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("point_cloud")
    parser.add_argument("--output")
    parser.add_argument("--min_actor_points", type=int, default=5)
    args = parser.parse_args()

    result = inspect(args.point_cloud, args.min_actor_points)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n")


if __name__ == "__main__":
    main()
