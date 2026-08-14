#!/usr/bin/env python3
"""Download an explicit subset of the locked AD-GS Waymo manifest."""

import argparse
import base64
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_PREFIX = "gs://waymo_open_dataset_v_1_4_1/individual_files"
EXPECTED_SCENES = (
    "scene006",
    "scene026",
    "scene090",
    "scene105",
    "scene108",
    "scene134",
    "scene150",
    "scene181",
)
MANIFEST_PATH = Path(__file__).parent / "manifests/waymo_adgs_8.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_base64(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_manifest(path=MANIFEST_PATH):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("bucket_prefix", "").rstrip("/") != EXPECTED_PREFIX:
        raise ValueError("manifest bucket prefix is not locked Waymo v1.4.1")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or len(sequences) != len(EXPECTED_SCENES):
        raise ValueError("manifest must contain exactly eight sequences")
    if tuple(item.get("scene") for item in sequences) != EXPECTED_SCENES:
        raise ValueError("manifest scene order does not match the locked AD-GS set")
    if len({item.get("gcs_object") for item in sequences}) != len(sequences):
        raise ValueError("manifest GCS objects are not unique")
    for item in sequences:
        gcs_object = item.get("gcs_object", "")
        filename = item.get("filename", "")
        if not gcs_object.startswith("validation/segment-") or not gcs_object.endswith(
            "_with_camera_labels.tfrecord"
        ):
            raise ValueError(f"unexpected GCS object for {item.get('scene')}")
        expected_filename = "individual_files_" + gcs_object.replace("/", "_")
        if filename != expected_filename:
            raise ValueError(f"local filename mismatch for {item.get('scene')}")
        if item.get("phase") not in {"diagnostic", "main_table_remaining"}:
            raise ValueError(f"invalid acquisition phase for {item.get('scene')}")
        first = item.get("first_frame")
        last = item.get("last_frame")
        if not isinstance(first, int) or not isinstance(last, int) or first > last:
            raise ValueError(f"invalid frame range for {item.get('scene')}")
    return manifest


def select_sequences(manifest, scene_ids):
    if not scene_ids:
        raise ValueError("at least one explicit --scene is required")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("duplicate scene IDs are not allowed")
    by_scene = {item["scene"]: item for item in manifest["sequences"]}
    unknown = [scene for scene in scene_ids if scene not in by_scene]
    if unknown:
        raise ValueError(f"unknown scene IDs: {unknown}")
    return [by_scene[scene] for scene in scene_ids]


def canonical_metadata(raw, sequence):
    name = raw.get("name", "")
    if not isinstance(name, str) or not name.endswith(sequence["gcs_object"]):
        raise ValueError(f"remote metadata name mismatch for {sequence['scene']}")
    try:
        size = int(raw["size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"remote size is invalid for {sequence['scene']}") from error
    if size <= 0:
        raise ValueError(f"remote size is not positive for {sequence['scene']}")
    return {
        "scene": sequence["scene"],
        "name": name,
        "size": size,
        "generation": raw.get("generation"),
        "etag": raw.get("etag"),
        "md5_hash": raw.get("md5Hash", raw.get("md5_hash")),
        "crc32c": raw.get("crc32c", raw.get("crc32c_hash")),
        "update_time": raw.get("updateTime", raw.get("update_time")),
    }


def validate_download(path, metadata):
    if not path.is_file():
        raise ValueError(f"download is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != metadata["size"]:
        raise ValueError(
            f"download size mismatch for {metadata['scene']}: "
            f"{actual_size} != {metadata['size']}"
        )
    expected_md5 = metadata.get("md5_hash")
    if expected_md5 and md5_base64(path) != expected_md5:
        raise ValueError(f"download MD5 mismatch for {metadata['scene']}")


def run_command(argv, command_records, *, redact_stdout=False):
    result = subprocess.run(argv, text=True, capture_output=True)
    command_records.append(
        {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": "<redacted>" if redact_stdout else result.stdout,
            "stderr": result.stderr,
        }
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(argv)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def acquire(scene_ids, destination, evidence_dir, gcloud):
    manifest = load_manifest()
    selected = select_sequences(manifest, scene_ids)
    destination = Path(destination).expanduser().resolve()
    evidence_dir = Path(evidence_dir).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    for name, path in (("destination", destination), ("evidence", evidence_dir)):
        if path == repository or repository in path.parents:
            raise ValueError(f"{name} directory must be outside the Git repository")
    if evidence_dir.exists():
        raise ValueError(f"evidence directory already exists: {evidence_dir}")
    evidence_dir.mkdir(parents=True)
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / ".gcs-staging"
    staging.mkdir(exist_ok=True)

    started = time.perf_counter()
    command_records = []
    (evidence_dir / "command.sh").write_text(
        shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        + "\n"
    )
    write_json(
        evidence_dir / "selected-manifest.json",
        {
            "bucket_prefix": manifest["bucket_prefix"],
            "manifest_path": str(MANIFEST_PATH.resolve()),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "sequences": selected,
        },
    )

    error = None
    receipts = []
    metadata_records = []
    try:
        run_command([gcloud, "auth", "print-access-token"], command_records, redact_stdout=True)
        version = run_command([gcloud, "--version"], command_records)
        (evidence_dir / "gcloud-version.txt").write_text(version)

        for sequence in selected:
            uri = f"{EXPECTED_PREFIX}/{sequence['gcs_object']}"
            stdout = run_command(
                [gcloud, "storage", "objects", "describe", uri, "--format=json"],
                command_records,
            )
            metadata = canonical_metadata(json.loads(stdout), sequence)
            metadata["uri"] = uri
            metadata_records.append(metadata)
        write_json(evidence_dir / "object-metadata.json", metadata_records)

        for sequence, metadata in zip(selected, metadata_records):
            final_path = destination / sequence["filename"]
            staged_path = staging / (Path(sequence["gcs_object"]).name + ".part")
            reused = False
            if final_path.exists():
                validate_download(final_path, metadata)
                reused = True
            else:
                if not staged_path.exists():
                    manifest_path = evidence_dir / f"gcloud-cp-{sequence['scene']}.csv"
                    run_command(
                        [
                            gcloud,
                            "storage",
                            "cp",
                            metadata["uri"],
                            str(staged_path),
                            f"--manifest-path={manifest_path}",
                        ],
                        command_records,
                    )
                validate_download(staged_path, metadata)
                os.replace(staged_path, final_path)
            validate_download(final_path, metadata)
            receipts.append(
                {
                    **metadata,
                    "first_frame": sequence["first_frame"],
                    "last_frame": sequence["last_frame"],
                    "local_path": str(final_path),
                    "local_sha256": sha256_file(final_path),
                    "reused_existing": reused,
                }
            )
        write_json(
            evidence_dir / "download-receipt.json",
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "objects": receipts,
            },
        )
        exitcode = 0
    except Exception as caught:
        error = caught
        exitcode = 2
        (evidence_dir / "error.log").write_text(traceback.format_exc())
    finally:
        write_json(evidence_dir / "commands.json", command_records)
        (evidence_dir / "exitcode.txt").write_text(f"{exitcode}\n")
        (evidence_dir / "wall_time_seconds.txt").write_text(
            f"{time.perf_counter() - started:.6f}\n"
        )
        artifact_lines = []
        for path in sorted(evidence_dir.iterdir()):
            if path.is_file() and path.name != "artifacts.sha256":
                artifact_lines.append(f"{sha256_file(path)}  {path.name}")
        (evidence_dir / "artifacts.sha256").write_text("\n".join(artifact_lines) + "\n")
    if error is not None:
        raise error
    return receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    gcloud = shutil.which("gcloud")
    if gcloud is None:
        parser.error("gcloud was not found on PATH")
    try:
        receipts = acquire(args.scene, args.destination, args.evidence_dir, gcloud)
    except Exception as error:
        print(f"DATA-003 failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": True, "objects": receipts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
