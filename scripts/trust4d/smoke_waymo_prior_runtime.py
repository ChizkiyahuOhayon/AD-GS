#!/usr/bin/env python3
"""Fail-closed smoke tests for the four pinned Waymo-prior runtimes."""

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "dpt": {"torch": "2.4.1", "torchvision": "0.19.1", "numpy": "1.26.4"},
    "sam": {
        "torch": "2.4.1",
        "torchvision": "0.19.1",
        "transformers": "4.46.3",
        "supervision": "0.22.0",
    },
    "flow": {
        "torch": "2.4.1",
        "torchvision": "0.19.1",
        "timm": "1.0.9",
        "einops": "0.8.0",
    },
}


def check_versions(stage, actual):
    mismatches = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in EXPECTED.get(stage, {}).items()
        if actual.get(name, "").split("+", 1)[0] != expected
    }
    if mismatches:
        raise ValueError(f"{stage} runtime version mismatch: {mismatches}")


def tensor_record(tensor):
    import torch

    if not torch.isfinite(tensor).all():
        raise ValueError("smoke output contains nonfinite values")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def gpu_contract(torch):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.version.cuda != "11.8":
        raise RuntimeError(f"expected PyTorch CUDA 11.8, got {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA A40":
        raise RuntimeError("physical GPU 0 must be an NVIDIA A40")
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda": torch.version.cuda,
    }


def smoke_dpt(root, checkpoint):
    import numpy as np
    import torch

    sys.path.insert(0, str(root))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    horizontal = np.linspace(0, 255, 84, dtype=np.uint8)[None, :, None]
    image = np.repeat(np.repeat(horizontal, 56, axis=0), 3, axis=2)
    depth = model.infer_image(image, input_size=56)
    if not np.isfinite(depth).all() or float(depth.max()) <= float(depth.min()):
        raise ValueError("DPT smoke depth must be finite and nonconstant")
    return {
        "depth": {
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "min": float(depth.min()),
            "max": float(depth.max()),
        }
    }


def smoke_sam(root, checkpoint):
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(root))
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.utils.misc import get_connected_components
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_model = build_sam2_video_predictor(config, str(checkpoint), device="cuda")
    del video_model
    torch.cuda.empty_cache()

    image_model = build_sam2(config, str(checkpoint), device="cuda")
    predictor = SAM2ImagePredictor(image_model)
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(96, dtype=np.uint8)[None]
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        box=np.array([[8, 8, 72, 48]], dtype=np.float32),
        multimask_output=False,
    )
    arrays = [torch.as_tensor(value) for value in (masks, scores, logits)]
    if not all(torch.isfinite(value).all() for value in arrays):
        raise ValueError("SAM image smoke output is nonfinite")

    labels, areas = get_connected_components(
        torch.tensor([[[[0, 1], [0, 1]]]], dtype=torch.uint8, device="cuda")
    )
    if labels.shape != areas.shape or labels.shape != (1, 1, 2, 2):
        raise ValueError("SAM connected-components extension returned wrong shapes")

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    grounding = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).cuda().eval()
    inputs = processor(images=Image.fromarray(image), text="car.", return_tensors="pt")
    inputs = inputs.to("cuda")
    with torch.inference_mode():
        outputs = grounding(**inputs)
    grounding_record = {
        "logits": tensor_record(outputs.logits),
        "pred_boxes": tensor_record(outputs.pred_boxes),
    }
    return {
        "sam_masks_shape": list(masks.shape),
        "sam_scores": [float(value) for value in np.asarray(scores).ravel()],
        "connected_components_shape": list(labels.shape),
        "grounding_dino": grounding_record,
    }


def smoke_flow(root):
    import torch

    model = torch.hub.load(
        "facebookresearch/co-tracker",
        "cotracker3_offline",
        trust_repo=True,
        skip_validation=True,
    ).cuda().eval()
    frames = torch.zeros((1, 4, 3, 64, 64), dtype=torch.float32, device="cuda")
    frames[:, :, 0] = torch.linspace(0, 255, 64, device="cuda")[None, None, None, :]
    queries = torch.tensor([[[0.0, 32.0, 32.0]]], device="cuda")
    with torch.inference_mode():
        tracks, visibility = model(frames, queries=queries)
    if tracks.shape[:3] != (1, 4, 1) or visibility.shape[:3] != (1, 4, 1):
        raise ValueError("CoTracker smoke output has the wrong sequence contract")
    return {"tracks": tensor_record(tracks), "visibility": tensor_record(visibility)}


def smoke_colmap():
    process = subprocess.run(
        ["colmap", "--version"], check=True, capture_output=True, text=True
    )
    version = (process.stdout + process.stderr).strip()
    if "COLMAP 3.7" not in version:
        raise ValueError(f"expected COLMAP 3.7, got: {version}")
    return {"version": version}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dpt", "sam", "flow", "colmap"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    actual = {
        name: importlib.metadata.version(name) for name in EXPECTED.get(args.stage, {})
    }
    check_versions(args.stage, actual)
    result = {"stage": args.stage, "versions": actual, "passed": True}

    if args.stage == "colmap":
        result.update(smoke_colmap())
    else:
        import torch

        result["gpu"] = gpu_contract(torch)
        if args.stage in {"dpt", "sam"} and args.checkpoint is None:
            raise ValueError(f"{args.stage} requires --checkpoint")
        if args.stage == "dpt":
            result.update(smoke_dpt(root, args.checkpoint.resolve()))
        elif args.stage == "sam":
            result.update(smoke_sam(root, args.checkpoint.resolve()))
        else:
            result.update(smoke_flow(root))
        result["peak_memory_allocated_mib"] = float(
            torch.cuda.max_memory_allocated() / 1024**2
        )
        if not math.isfinite(result["peak_memory_allocated_mib"]):
            raise ValueError("invalid peak GPU memory")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
