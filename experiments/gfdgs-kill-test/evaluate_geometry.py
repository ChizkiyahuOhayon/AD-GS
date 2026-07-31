"""Evaluate paired sparse depth and actor contact for the locked G2 gate."""

import argparse
import gc
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_config
from gaussian_renderer import render
from models.geometry_metrics import paired_sparse_depth_metrics
from scene import Scene
from scene.env import EnvironmentMap
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--oracle_model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="./arguments/waymo.py")
    return parser.parse_args()


def load_scene(source, model_path, iteration, config_path, oracle_contact):
    parser = ArgumentParser(add_help=False)
    config = get_config(config_path)
    model_params = ModelParams(parser, config)
    pipeline_params = PipelineParams(parser, config)
    parsed = parser.parse_args(
        ["-s", source, "-m", model_path, "--data_device", "cuda:0"]
    )
    dataset = model_params.extract(parsed)
    dataset.oracle_contact = oracle_contact
    pipeline = pipeline_params.extract(parsed)
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.order_args,
        oracle_contact=oracle_contact,
    )
    env_map = EnvironmentMap(**dataset.env_args)
    scene = Scene(
        dataset,
        gaussians,
        env_map,
        load_iteration=iteration,
        shuffle=False,
    )
    if not oracle_contact:
        gaussians.configure_oracle_contact(str(Path(source) / "points3d.ply"))
    return scene, pipeline


def evaluate_arm(source, model_path, iteration, config, samples_dir, oracle_contact):
    scene, pipeline = load_scene(
        source, model_path, iteration, config, oracle_contact
    )
    sample_depth = []
    sample_actor_id = []
    inverse_depth = []
    opacity = []
    contact_before = []
    contact_after = []
    contact_pair_count = 0

    with torch.no_grad():
        for view in scene.getTestCameras():
            sample_path = samples_dir / (Path(view.image_name).stem + ".npz")
            if not sample_path.exists():
                raise FileNotFoundError(sample_path)
            samples = np.load(sample_path)
            package = render(view, scene.gaussians, scene.env_map, pipeline)
            rendered_inverse = package["depth"].detach().cpu().numpy()
            rendered_opacity = package["img_opacity"].detach().cpu().numpy()
            expected_shape = tuple(int(value) for value in samples["image_shape"])
            if rendered_inverse.shape != expected_shape or rendered_opacity.shape != expected_shape:
                raise ValueError("render/sample shape mismatch for {}".format(view.image_name))
            u = samples["u"]
            v = samples["v"]
            sample_depth.append(samples["depth"])
            sample_actor_id.append(samples["actor_id"])
            inverse_depth.append(rendered_inverse[v, u])
            opacity.append(rendered_opacity[v, u])

            original_flag = scene.gaussians.oracle_contact
            scene.gaussians.oracle_contact = True
            _, diagnostics = scene.gaussians._get_deformed_xyz_and_contact(view.time)
            scene.gaussians.oracle_contact = original_flag
            contact_before.append(float(diagnostics["mean_abs_before"].detach().cpu()))
            contact_after.append(float(diagnostics["mean_abs_after"].detach().cpu()))
            contact_pair_count += int(diagnostics["actor_count"])

    track_ids = sorted(int(actor_id) for actor_id in scene.gaussians.oracle_contact_tracks)
    result = {
        "ground_truth_depth": np.concatenate(sample_depth),
        "actor_id": np.concatenate(sample_actor_id),
        "inverse_depth": np.concatenate(inverse_depth),
        "opacity": np.concatenate(opacity),
        "contact_mean_abs_before": float(np.mean(contact_before)),
        "contact_mean_abs_after": float(np.mean(contact_after)),
        "contact_actor_time_pair_count": contact_pair_count,
        "contact_actor_ids": track_ids,
        "test_view_count": len(scene.getTestCameras()),
    }
    del scene
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    output_path = Path(args.output)
    sample_output = output_path.with_suffix(".npz")
    if output_path.exists() or sample_output.exists():
        raise FileExistsError("refusing to overwrite geometry evaluation output")

    torch.cuda.set_device("cuda:0")
    safe_state(True)
    samples_dir = Path(args.samples)
    baseline = evaluate_arm(
        args.source,
        args.baseline_model,
        args.iteration,
        args.config,
        samples_dir,
        oracle_contact=False,
    )
    oracle = evaluate_arm(
        args.source,
        args.oracle_model,
        args.iteration,
        args.config,
        samples_dir,
        oracle_contact=True,
    )

    if not np.array_equal(baseline["ground_truth_depth"], oracle["ground_truth_depth"]):
        raise ValueError("arms did not use identical ground-truth depth samples")
    if not np.array_equal(baseline["actor_id"], oracle["actor_id"]):
        raise ValueError("arms did not use identical actor samples")
    if baseline["contact_actor_ids"] != oracle["contact_actor_ids"]:
        raise ValueError("arms did not use identical contact actors")
    if baseline["contact_actor_time_pair_count"] != oracle["contact_actor_time_pair_count"]:
        raise ValueError("arms did not use identical actor-time pairs")

    depth = paired_sparse_depth_metrics(
        baseline["ground_truth_depth"],
        baseline["actor_id"],
        baseline["inverse_depth"],
        baseline["opacity"],
        oracle["inverse_depth"],
        oracle["opacity"],
        min_opacity=0.1,
        min_pixels=100,
        min_actors=5,
    )
    baseline_contact = baseline["contact_mean_abs_before"]
    oracle_contact = oracle["contact_mean_abs_after"]
    contact_improvement = (baseline_contact - oracle_contact) / baseline_contact
    contact = {
        "actor_ids": baseline["contact_actor_ids"],
        "actor_time_pair_count": baseline["contact_actor_time_pair_count"],
        "baseline_mean_abs": baseline_contact,
        "oracle_mean_abs": oracle_contact,
        "oracle_unprojected_mean_abs": oracle["contact_mean_abs_before"],
        "relative_improvement": float(contact_improvement),
    }
    result = {
        "iteration": args.iteration,
        "test_view_count": baseline["test_view_count"],
        "raw_lidar_sample_count": int(baseline["ground_truth_depth"].size),
        "depth": depth,
        "contact": contact,
        "g2_geometry_gate_pass": bool(
            depth["relative_improvement"] >= 0.10
            and contact["relative_improvement"] >= 0.50
        ),
    }
    np.savez_compressed(
        sample_output,
        ground_truth_depth=baseline["ground_truth_depth"],
        actor_id=baseline["actor_id"],
        baseline_inverse_depth=baseline["inverse_depth"],
        baseline_opacity=baseline["opacity"],
        oracle_inverse_depth=oracle["inverse_depth"],
        oracle_opacity=oracle["opacity"],
    )
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
