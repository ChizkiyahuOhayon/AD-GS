#!/usr/bin/env python3
"""Cache the immutable Grounding DINO snapshot used by released semantic.py."""

import argparse
import hashlib
import json
from pathlib import Path


REPO_ID = "IDEA-Research/grounding-dino-base"
REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
MODEL_SHA256 = "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"
MODEL_SIZE = 933400872
FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    hf_home = args.hf_home.expanduser().resolve()
    hub_cache = hf_home / "hub"
    snapshot = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            cache_dir=hub_cache,
            allow_patterns=list(FILES),
        )
    ).resolve()
    if snapshot.name != REVISION:
        raise ValueError(f"unexpected snapshot revision: {snapshot}")

    records = {}
    for filename in FILES:
        path = snapshot / filename
        if not path.is_file():
            raise ValueError(f"snapshot is missing {filename}")
        records[filename] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    model = records["model.safetensors"]
    if model["size_bytes"] != MODEL_SIZE or model["sha256"] != MODEL_SHA256:
        raise ValueError(f"Grounding DINO model contract failed: {model}")

    repo_cache = snapshot.parent.parent
    refs = repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(REVISION + "\n")

    result = {
        "passed": True,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot": str(snapshot),
        "files": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
