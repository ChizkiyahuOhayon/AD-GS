import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_PREFIX = "gs://waymo_open_dataset_v_1_4_1/individual_files"


def load_objects(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    prefix = manifest["bucket_prefix"].rstrip("/")
    if prefix != EXPECTED_PREFIX:
        raise ValueError("refusing non-v1.4.1 bucket prefix: {}".format(prefix))

    objects = []
    for sequence in manifest["sequences"]:
        filename = sequence["filename"]
        gcs_object = sequence["gcs_object"]
        if not gcs_object.startswith("validation/segment-"):
            raise ValueError("refusing unexpected GCS object: {}".format(gcs_object))
        objects.append(
            {
                "scene": sequence["scene"],
                "filename": filename,
                "gcs_object": gcs_object,
                "download_name": Path(gcs_object).name,
                "uri": "{}/{}".format(prefix, gcs_object),
            }
        )
    if len(objects) != 8 or len({item["uri"] for item in objects}) != 8:
        raise ValueError("expected exactly eight unique Waymo objects")
    return objects


def run_json(command, attempts=3):
    last_error = None
    for attempt in range(attempts):
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        last_error = RuntimeError(
            "command failed with code {}: {}".format(
                result.returncode, result.stderr.strip()
            )
        )
        if attempt + 1 < attempts:
            time.sleep(2 ** attempt)
    raise last_error


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary_path), str(path))


def main():
    parser = argparse.ArgumentParser(
        description="Download the exact AD-GS Waymo v1.4.1 manifest with evidence."
    )
    parser.add_argument("manifest")
    parser.add_argument("destination")
    parser.add_argument("evidence_dir")
    parser.add_argument("--gcloud", required=True)
    parser.add_argument("--access-token-file", required=True)
    parser.add_argument("--parallel", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    token_path = Path(args.access_token_file)
    if not token_path.is_file():
        raise FileNotFoundError("access token file not found")

    destination = Path(args.destination)
    evidence_dir = Path(args.evidence_dir)
    destination.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    objects = load_objects(args.manifest)

    metadata = []
    for item in objects:
        record = run_json(
            [
                args.gcloud,
                "storage",
                "objects",
                "describe",
                item["uri"],
                "--format=json",
                "--access-token-file={}".format(token_path),
            ]
        )
        if not record.get("name", "").endswith(item["gcs_object"]):
            raise ValueError("metadata name mismatch for {}".format(item["scene"]))
        metadata.append({"scene": item["scene"], "uri": item["uri"], **record})
    write_json(evidence_dir / "object-metadata.json", metadata)
    if args.metadata_only:
        return

    staging = destination / ".gcs-staging"
    staging.mkdir(exist_ok=True)
    pending = []
    for item, record in zip(objects, metadata):
        local_path = destination / item["filename"]
        expected_size = int(record["size"])
        if local_path.exists():
            if local_path.stat().st_size != expected_size:
                raise ValueError("existing final file has wrong size: {}".format(local_path))
        else:
            pending.append(item)
    if pending:
        def copy_one(item):
            command = [
                args.gcloud,
                "storage",
                "cp",
                item["uri"],
                str(staging / item["download_name"]),
                "--manifest-path={}".format(
                    evidence_dir / "gcloud-cp-manifest-{}.csv".format(item["scene"])
                ),
                "--access-token-file={}".format(token_path),
            ]
            environment = os.environ.copy()
            environment["CLOUDSDK_STORAGE_PROCESS_COUNT"] = "1"
            environment["CLOUDSDK_STORAGE_THREAD_COUNT"] = "4"
            environment["CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD"] = "50Mi"
            environment["CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_COMPONENT_SIZE"] = "64Mi"
            environment["CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_MAX_COMPONENTS"] = "4"
            subprocess.run(command, check=True, env=environment)

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            list(executor.map(copy_one, pending))
        metadata_by_scene = {record["scene"]: record for record in metadata}
        for item in pending:
            staged_path = staging / item["download_name"]
            expected_size = int(metadata_by_scene[item["scene"]]["size"])
            if staged_path.stat().st_size != expected_size:
                raise ValueError("staged file has wrong size: {}".format(staged_path))
            os.replace(str(staged_path), str(destination / item["filename"]))

    receipts = []
    for item, record in zip(objects, metadata):
        local_path = destination / item["filename"]
        actual_size = local_path.stat().st_size
        expected_size = int(record["size"])
        if actual_size != expected_size:
            raise ValueError(
                "size mismatch for {}: {} != {}".format(
                    item["scene"], actual_size, expected_size
                )
            )
        receipts.append(
            {
                "scene": item["scene"],
                "uri": item["uri"],
                "gcs_object": item["gcs_object"],
                "local_path": str(local_path.resolve()),
                "size": actual_size,
                "generation": record.get("generation"),
                "etag": record.get("etag"),
                "md5_hash": record.get("md5_hash", record.get("md5Hash")),
                "crc32c_hash": record.get("crc32c_hash", record.get("crc32c")),
                "local_sha256": sha256_file(local_path),
            }
        )
    write_json(
        evidence_dir / "download-receipt.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "parallel_threads": args.parallel,
            "objects": receipts,
        },
    )


if __name__ == "__main__":
    main()
