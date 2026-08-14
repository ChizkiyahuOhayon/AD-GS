import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.trust4d.download_waymo_manifest import (
    EXPECTED_PREFIX,
    EXPECTED_SCENES,
    MANIFEST_PATH,
    load_manifest,
    select_sequences,
    validate_download,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/trust4d/download_waymo_manifest.py"


class DownloadWaymoManifestTest(unittest.TestCase):
    def test_manifest_matches_official_preprocess_commands(self):
        manifest = load_manifest()
        pattern = re.compile(
            r"python scripts/waymo/waymo\.py \$1/(?P<filename>\S+) "
            r"\./data/waymo/(?P<scene>scene\d+) "
            r"--first_frame (?P<first>\d+) --last_frame (?P<last>\d+) --use_color"
        )
        official = [
            {
                "scene": match.group("scene"),
                "filename": match.group("filename"),
                "first_frame": int(match.group("first")),
                "last_frame": int(match.group("last")),
            }
            for match in pattern.finditer(
                (REPOSITORY / "scripts/waymo/prepare-waymo.sh").read_text()
            )
        ]
        recorded = [
            {key: item[key] for key in ("scene", "filename", "first_frame", "last_frame")}
            for item in manifest["sequences"]
        ]
        self.assertEqual(recorded, official)
        self.assertEqual(tuple(item["scene"] for item in manifest["sequences"]), EXPECTED_SCENES)

    def test_explicit_selection_rejects_unknown_and_duplicates(self):
        manifest = load_manifest()
        selected = select_sequences(manifest, ["scene006", "scene026", "scene090"])
        self.assertEqual([item["scene"] for item in selected], ["scene006", "scene026", "scene090"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_sequences(manifest, ["scene006", "scene006"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            select_sequences(manifest, ["scene999"])

    def test_validate_download_checks_size_and_md5(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "record.tfrecord"
            path.write_bytes(b"abc")
            metadata = {
                "scene": "scene006",
                "size": 3,
                "md5_hash": base64.b64encode(hashlib.md5(b"abc").digest()).decode(),
            }
            validate_download(path, metadata)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                validate_download(path, metadata)

    def test_fake_gcloud_downloads_exact_two_objects_with_evidence(self):
        with tempfile.TemporaryDirectory() as root_string:
            root = Path(root_string)
            fake_bin = root / "bin"
            destination = root / "raw"
            evidence = root / "evidence"
            log = root / "gcloud.log"
            fake_bin.mkdir()
            fake_gcloud = fake_bin / "gcloud"
            fake_gcloud.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$TEST_GCLOUD_LOG\"\n"
                "if [[ \"$1\" == '--version' ]]; then\n"
                "  printf 'Google Cloud SDK test\\n'\n"
                "elif [[ \"$1 ${2:-}\" == 'auth print-access-token' ]]; then\n"
                "  printf 'secret-test-token\\n'\n"
                "elif [[ \"$1 ${2:-} ${3:-}\" == 'storage objects describe' ]]; then\n"
                "  name=${4#gs://}\n"
                "  printf '{\"name\":\"%s\",\"size\":\"3\",\"md5Hash\":\"kAFQmDzST7DWlj99KOF/cg==\"}\\n' \"$name\"\n"
                "elif [[ \"$1 ${2:-}\" == 'storage cp' ]]; then\n"
                "  printf 'abc' > \"$4\"\n"
                "else\n"
                "  exit 64\n"
                "fi\n"
            )
            fake_gcloud.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["TEST_GCLOUD_LOG"] = str(log)
            command = [
                sys.executable,
                str(SCRIPT),
                "--scene",
                "scene006",
                "--scene",
                "scene026",
                "--destination",
                str(destination),
                "--evidence-dir",
                str(evidence),
            ]
            subprocess.run(command, cwd=REPOSITORY, env=environment, check=True, capture_output=True, text=True)

            manifest = load_manifest(MANIFEST_PATH)
            selected = select_sequences(manifest, ["scene006", "scene026"])
            for item in selected:
                self.assertEqual((destination / item["filename"]).read_bytes(), b"abc")
            calls = log.read_text()
            for item in selected:
                self.assertIn(f"{EXPECTED_PREFIX}/{item['gcs_object']}", calls)
            self.assertNotIn("/individual_files/individual_files_validation_segment-", calls)
            self.assertEqual((evidence / "exitcode.txt").read_text().strip(), "0")
            receipt = json.loads((evidence / "download-receipt.json").read_text())
            self.assertEqual([item["scene"] for item in receipt["objects"]], ["scene006", "scene026"])
            self.assertNotIn("secret-test-token", (evidence / "commands.json").read_text())
            self.assertTrue((evidence / "artifacts.sha256").read_text().strip())


if __name__ == "__main__":
    unittest.main()
