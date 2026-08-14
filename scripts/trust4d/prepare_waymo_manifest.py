#!/usr/bin/env python3
"""Preprocess an explicit downloaded subset with the unchanged AD-GS converter."""

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    from .audit_official_source import audit_repository
    from .download_waymo_manifest import (
        EXPECTED_PREFIX,
        MANIFEST_PATH,
        load_manifest,
        select_sequences,
        sha256_file,
        write_json,
    )
except ImportError:
    from audit_official_source import audit_repository
    from download_waymo_manifest import (
        EXPECTED_PREFIX,
        MANIFEST_PATH,
        load_manifest,
        select_sequences,
        sha256_file,
        write_json,
    )


ENVIRONMENT_NAME = "trust4d-waymo-prep"
EXPECTED_PREPROCESSOR_SHA256 = (
    "f87905fb7c867d572679a6b7ea92dbe4b085d5a4a695f70ee1776c2058188bd6"
)


def validate_download_receipt(download_evidence, selected, raw_dir):
    download_evidence = Path(download_evidence).expanduser().resolve()
    raw_dir = Path(raw_dir).expanduser().resolve()
    exitcode_path = download_evidence / "exitcode.txt"
    receipt_path = download_evidence / "download-receipt.json"
    if not exitcode_path.is_file() or exitcode_path.read_text().strip() != "0":
        raise ValueError("download evidence is missing or did not pass")
    if not receipt_path.is_file():
        raise ValueError("download receipt is missing")
    receipt = json.loads(receipt_path.read_text())
    objects = receipt.get("objects")
    if not isinstance(objects, list):
        raise ValueError("download receipt objects are malformed")
    by_scene = {}
    for record in objects:
        scene = record.get("scene")
        if scene in by_scene:
            raise ValueError(f"duplicate download receipt for {scene}")
        by_scene[scene] = record

    validated = []
    for sequence in selected:
        scene = sequence["scene"]
        record = by_scene.get(scene)
        if record is None:
            raise ValueError(f"download receipt is missing {scene}")
        raw_path = (raw_dir / sequence["filename"]).resolve()
        expected_uri = f"{EXPECTED_PREFIX}/{sequence['gcs_object']}"
        if Path(record.get("local_path", "")).expanduser().resolve() != raw_path:
            raise ValueError(f"download receipt local path mismatch for {scene}")
        if record.get("uri") != expected_uri:
            raise ValueError(f"download receipt URI mismatch for {scene}")
        if not raw_path.is_file():
            raise ValueError(f"raw TFRecord is missing for {scene}: {raw_path}")
        if record.get("size") != raw_path.stat().st_size:
            raise ValueError(f"download receipt size mismatch for {scene}")
        actual_sha = sha256_file(raw_path)
        if record.get("local_sha256") != actual_sha:
            raise ValueError(f"download receipt SHA-256 mismatch for {scene}")
        validated.append(
            {
                "scene": scene,
                "path": str(raw_path),
                "size_bytes": raw_path.stat().st_size,
                "sha256": actual_sha,
                "download_receipt": record,
            }
        )
    return {
        "download_evidence": str(download_evidence),
        "download_receipt_sha256": sha256_file(receipt_path),
        "objects": validated,
    }


def build_scene_commands(conda, repository, raw_record, processed_scene, sequence, evidence):
    frame_count = sequence["last_frame"] - sequence["first_frame"] + 1
    preprocess = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        ENVIRONMENT_NAME,
        "python",
        str(repository / "scripts/waymo/waymo.py"),
        raw_record["path"],
        str(processed_scene),
        "--first_frame",
        str(sequence["first_frame"]),
        "--last_frame",
        str(sequence["last_frame"]),
        "--use_color",
    ]
    validate = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        ENVIRONMENT_NAME,
        "python",
        str(repository / "scripts/trust4d/validate_waymo_scene006.py"),
        "--scene",
        str(processed_scene),
        "--expected-frame-count",
        str(frame_count),
        "--output",
        str(evidence / f"{sequence['scene']}.validation.json"),
    ]
    return preprocess, validate


def run_logged(argv, stdout_path, stderr_path, environment):
    started = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr, env=environment)
    return result.returncode, time.perf_counter() - started


def finalize_evidence(evidence, exitcode, started, command_records):
    write_json(evidence / "commands.json", command_records)
    (evidence / "exitcode.txt").write_text(f"{exitcode}\n")
    (evidence / "wall_time_seconds.txt").write_text(
        f"{time.perf_counter() - started:.6f}\n"
    )
    lines = []
    for path in sorted(evidence.iterdir()):
        if path.is_file() and path.name != "artifacts.sha256":
            lines.append(f"{sha256_file(path)}  {path.name}")
    (evidence / "artifacts.sha256").write_text("\n".join(lines) + "\n")


def prepare(scene_ids, raw_dir, download_evidence, processed_root, evidence, conda):
    manifest = load_manifest()
    selected = select_sequences(manifest, scene_ids)
    repository = Path(__file__).resolve().parents[2]
    raw_dir = Path(raw_dir).expanduser().resolve()
    processed_root = Path(processed_root).expanduser().resolve()
    evidence = Path(evidence).expanduser().resolve()
    download_evidence = Path(download_evidence).expanduser().resolve()
    for name, path in (
        ("raw", raw_dir),
        ("processed", processed_root),
        ("download evidence", download_evidence),
        ("preprocess evidence", evidence),
    ):
        if path == repository or repository in path.parents:
            raise ValueError(f"{name} path must be outside the Git repository")
    if evidence.exists():
        raise ValueError(f"preprocessing evidence path must be new: {evidence}")
    processed_scenes = {item["scene"]: processed_root / item["scene"] for item in selected}
    existing = [scene for scene, path in processed_scenes.items() if path.exists()]
    if existing:
        raise ValueError(f"processed scene paths must be new: {existing}")

    source_audit = audit_repository(repository)
    preprocessor = repository / "scripts/waymo/waymo.py"
    if sha256_file(preprocessor) != EXPECTED_PREPROCESSOR_SHA256:
        raise ValueError("official Waymo preprocessor SHA-256 mismatch")
    env_list = subprocess.run(
        [conda, "env", "list", "--json"], check=True, text=True, capture_output=True
    )
    environments = json.loads(env_list.stdout).get("envs", [])
    if ENVIRONMENT_NAME not in {Path(path).name for path in environments}:
        raise ValueError(f"required Conda environment is missing: {ENVIRONMENT_NAME}")
    raw_contract = validate_download_receipt(download_evidence, selected, raw_dir)
    raw_by_scene = {item["scene"]: item for item in raw_contract["objects"]}
    environment_freeze = subprocess.run(
        [conda, "run", "-n", ENVIRONMENT_NAME, "python", "-m", "pip", "freeze"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout

    evidence.mkdir(parents=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command_records = []
    (evidence / "command.sh").write_text(
        shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        + "\n"
    )
    write_json(evidence / "official_source_audit.json", source_audit)
    write_json(evidence / "raw_input_contract.json", raw_contract)
    write_json(
        evidence / "selected-manifest.json",
        {
            "manifest_path": str(MANIFEST_PATH.resolve()),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "sequences": selected,
        },
    )
    (evidence / "environment.txt").write_text(environment_freeze)
    write_json(
        evidence / "host.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
        },
    )

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    validations = []
    exitcode = 0
    try:
        for sequence in selected:
            scene = sequence["scene"]
            processed_scene = processed_scenes[scene]
            preprocess, validate = build_scene_commands(
                conda,
                repository,
                raw_by_scene[scene],
                processed_scene,
                sequence,
                evidence,
            )
            for stage, argv in (("preprocess", preprocess), ("validate", validate)):
                label = f"{scene}.{stage}"
                command_record = {"stage": label, "argv": argv}
                command_records.append(command_record)
                code, elapsed = run_logged(
                    argv,
                    evidence / f"{label}.stdout.log",
                    evidence / f"{label}.stderr.log",
                    environment,
                )
                command_record.update(returncode=code, wall_time_seconds=elapsed)
                (evidence / f"{label}.exitcode.txt").write_text(f"{code}\n")
                (evidence / f"{label}.wall_time_seconds.txt").write_text(
                    f"{elapsed:.6f}\n"
                )
                if code != 0:
                    (evidence / "failed_stage.txt").write_text(label + "\n")
                    raise RuntimeError(f"{label} failed with exit code {code}")
            validation_path = evidence / f"{scene}.validation.json"
            validation = json.loads(validation_path.read_text())
            if validation.get("passed") is not True:
                raise ValueError(f"{scene} validation did not report passed=true")
            validations.append(
                {
                    "scene": scene,
                    "processed_path": str(processed_scene.resolve()),
                    "frame_count": validation["frame_count"],
                    "validation_sha256": sha256_file(validation_path),
                }
            )
        write_json(
            evidence / "preprocessing-receipt.json",
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "passed": True,
                "scenes": validations,
            },
        )
    except Exception:
        exitcode = 2
        (evidence / "error.log").write_text(traceback.format_exc())
    finally:
        finalize_evidence(evidence, exitcode, started, command_records)
    if exitcode != 0:
        raise RuntimeError((evidence / "error.log").read_text().splitlines()[-1])
    return validations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--download-evidence", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    conda = shutil.which("conda")
    if conda is None:
        parser.error("conda was not found on PATH")
    try:
        result = prepare(
            args.scene,
            args.raw_dir,
            args.download_evidence,
            args.processed_root,
            args.evidence_dir,
            conda,
        )
    except Exception as error:
        print(f"DATA-003 preprocessing failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": True, "scenes": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
