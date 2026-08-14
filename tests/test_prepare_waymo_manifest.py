import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.trust4d.download_waymo_manifest import load_manifest, select_sequences
from scripts.trust4d.prepare_waymo_manifest import (
    build_scene_commands,
    validate_download_receipt,
)


class PrepareWaymoManifestTest(unittest.TestCase):
    def make_download(self, root, scenes):
        root = Path(root)
        raw = root / "raw"
        evidence = root / "download-evidence"
        raw.mkdir()
        evidence.mkdir()
        selected = select_sequences(load_manifest(), scenes)
        records = []
        for index, item in enumerate(selected):
            payload = f"record-{index}".encode()
            path = raw / item["filename"]
            path.write_bytes(payload)
            records.append(
                {
                    "scene": item["scene"],
                    "uri": (
                        "gs://waymo_open_dataset_v_1_4_1/individual_files/"
                        + item["gcs_object"]
                    ),
                    "local_path": str(path.resolve()),
                    "size": len(payload),
                    "local_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        (evidence / "exitcode.txt").write_text("0\n")
        (evidence / "download-receipt.json").write_text(
            json.dumps({"objects": records}) + "\n"
        )
        return raw, evidence, selected

    def test_receipt_rechecks_raw_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            raw, evidence, selected = self.make_download(root, ["scene026", "scene090"])
            result = validate_download_receipt(evidence, selected, raw)
            self.assertEqual([item["scene"] for item in result["objects"]], ["scene026", "scene090"])
            (raw / selected[0]["filename"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                validate_download_receipt(evidence, selected, raw)

    def test_receipt_rejects_wrong_remote_uri(self):
        with tempfile.TemporaryDirectory() as root:
            raw, evidence, selected = self.make_download(root, ["scene026"])
            receipt_path = evidence / "download-receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["objects"][0]["uri"] = "gs://wrong/object"
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "URI mismatch"):
                validate_download_receipt(evidence, selected, raw)

    def test_commands_use_complete_official_frame_range(self):
        sequence = select_sequences(load_manifest(), ["scene090"])[0]
        preprocess, validate = build_scene_commands(
            "/opt/conda/bin/conda",
            Path("/repo"),
            {"path": "/raw/scene090.tfrecord"},
            Path("/processed/scene090"),
            sequence,
            Path("/evidence"),
        )
        self.assertEqual(preprocess[preprocess.index("--first_frame") + 1], "0")
        self.assertEqual(preprocess[preprocess.index("--last_frame") + 1], "102")
        self.assertIn("--use_color", preprocess)
        self.assertEqual(
            validate[validate.index("--expected-frame-count") + 1], "103"
        )

    def test_receipt_requires_successful_download(self):
        with tempfile.TemporaryDirectory() as root:
            raw, evidence, selected = self.make_download(root, ["scene026"])
            (evidence / "exitcode.txt").write_text("2\n")
            with self.assertRaisesRegex(ValueError, "did not pass"):
                validate_download_receipt(evidence, selected, raw)


if __name__ == "__main__":
    unittest.main()
