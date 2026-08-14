#!/usr/bin/env python3
"""Pin, cache, and execute the released AD-GS LPIPS evaluator weights."""

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.request
from pathlib import Path


LPIPS_COMMIT = "082bb24f84c091ea94de2867d34c4544f68e0963"
LPIPS_FILES = {
    "alex.pth": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    "vgg.pth": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
}
BACKBONE_FILES = {
    "alexnet": ("alexnet-owt-7be5be79.pth", "7be5be79"),
    "vgg16": ("vgg16-397923af.pth", "397923af"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_digest(path, expected_digest):
    actual = sha256(path)
    if actual != expected_digest:
        raise ValueError(
            f"checksum mismatch for {Path(path).name}: expected {expected_digest}, got {actual}"
        )
    return actual


def ensure_lpips_file(cache_dir, filename, expected_digest):
    destination = cache_dir / filename
    if destination.is_file():
        validate_digest(destination, expected_digest)
        return destination

    url = (
        "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
        f"{LPIPS_COMMIT}/lpips/weights/v0.1/{filename}"
    )
    temporary = destination.with_name(f".{filename}.{os.getpid()}.download")
    try:
        urllib.request.urlretrieve(url, temporary)
        validate_digest(temporary, expected_digest)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def file_record(path):
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adgs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    adgs_root = args.adgs_root.expanduser().resolve()
    sys.path.insert(0, str(adgs_root))

    import torch
    from torchvision import models

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA A40":
        raise RuntimeError("the evaluator smoke test requires physical GPU 0 to be an A40")

    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    backbone_records = {}
    for name, (filename, digest_prefix) in BACKBONE_FILES.items():
        path = cache_dir / filename
        if not path.is_file():
            raise ValueError(f"torchvision did not cache {filename}")
        record = file_record(path)
        if not record["sha256"].startswith(digest_prefix):
            raise ValueError(f"torchvision checksum prefix failed for {filename}")
        backbone_records[name] = record

    lpips_records = {
        filename: file_record(ensure_lpips_file(cache_dir, filename, expected))
        for filename, expected in LPIPS_FILES.items()
    }

    from lpipsPyTorch import lpips

    first = torch.zeros((1, 3, 64, 64), dtype=torch.float32, device="cuda")
    second = torch.ones_like(first)
    scores = {}
    with torch.inference_mode():
        for net_type in ("alex", "vgg"):
            score = float(lpips(first, second, net_type=net_type).item())
            if not math.isfinite(score) or score < 0:
                raise ValueError(f"LPIPS({net_type}) smoke score is invalid: {score}")
            scores[net_type] = score
            torch.cuda.empty_cache()

    result = {
        "passed": True,
        "lpips_commit": LPIPS_COMMIT,
        "torch_hub_cache": str(cache_dir.resolve()),
        "backbones": backbone_records,
        "lpips_weights": lpips_records,
        "smoke_scores": scores,
        "gpu": torch.cuda.get_device_name(0),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
