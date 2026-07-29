import argparse
import hashlib
import json
import os
import subprocess
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
        objects.append(
            {
                "scene": sequence["scene"],
                "filename": filename,
                "uri": "{}/{}".format(prefix, filename),
            }
        )
    if len(objects) != 8 or len({item["uri"] for item in objects}) != 8:
        raise ValueError("expected exactly eight unique Waymo objects")
    return objects


def run_json(command):
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


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
    parser.add_argument("--parallel", type=int, choices=range(1, 5), default=4)
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
        if not record.get("name", "").endswith(item["filename"]):
            raise ValueError("metadata name mismatch for {}".format(item["scene"]))
        metadata.append({"scene": item["scene"], "uri": item["uri"], **record})
    write_json(evidence_dir / "object-metadata.json", metadata)

    copy_manifest = evidence_dir / "gcloud-cp-manifest.csv"
    command = [args.gcloud, "storage", "cp"]
    command.extend(item["uri"] for item in objects)
    command.extend(
        [
            str(destination),
            "--manifest-path={}".format(copy_manifest),
            "--access-token-file={}".format(token_path),
        ]
    )
    environment = os.environ.copy()
    environment["CLOUDSDK_STORAGE_PROCESS_COUNT"] = "1"
    environment["CLOUDSDK_STORAGE_THREAD_COUNT"] = str(args.parallel)
    subprocess.run(command, check=True, env=environment)

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
                "local_path": str(local_path.resolve()),
                "size": actual_size,
                "generation": record.get("generation"),
                "etag": record.get("etag"),
                "md5Hash": record.get("md5Hash"),
                "crc32c": record.get("crc32c"),
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
