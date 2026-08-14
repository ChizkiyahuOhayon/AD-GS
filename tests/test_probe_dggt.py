import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts.trust4d.probe_dggt import (
    EXPECTED_DGGT_COMMIT,
    evaluate_gate,
    load_verified_selection,
    tensor_summary,
    verify_checkpoint,
    verify_source_contract,
)
from scripts.trust4d.select_waymo_training_frames import select_training_frames


def valid_predictions():
    return {
        "pose_enc": torch.zeros(1, 4, 9),
        "world_points": torch.zeros(1, 4, 2, 3),
        "world_points_conf": torch.ones(1, 4, 2),
        "gs_map": torch.zeros(1, 4, 2, 12),
        "gs_conf": torch.ones(1, 4, 2),
        "dynamic_conf": torch.zeros(1, 4, 2, 1),
        "depth": torch.ones(1, 4, 2, 1),
        "depth_conf": torch.ones(1, 4, 2, 1),
        "semantic_logits": torch.zeros(1, 4, 2, 11),
        "nested": [torch.ones(1)],
    }


class ProbeDggtTest(unittest.TestCase):
    def make_selection(self, root):
        scene = Path(root) / "scene006"
        image_dir = scene / "image"
        image_dir.mkdir(parents=True)
        for index in range(5):
            (image_dir / f"{index:06d}.jpg").write_bytes(bytes([index]))
        np.savez(
            scene / "cameras.npz",
            is_val_list=np.asarray([False, False, True, False, False]),
            time_stamps=np.arange(5, dtype=np.float32),
        )
        selection_path = Path(root) / "selection.json"
        selection_path.write_text(
            json.dumps(select_training_frames(scene, count=4), indent=2, sort_keys=True)
            + "\n"
        )
        return scene, selection_path

    def test_valid_contract_passes(self):
        gate = evaluate_gate(valid_predictions(), peak_allocated_mib=1024)
        self.assertTrue(gate["passed"])

    def test_missing_required_key_fails(self):
        predictions = valid_predictions()
        del predictions["depth"]
        gate = evaluate_gate(predictions, peak_allocated_mib=1024)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["missing_required_keys"], ["depth"])

    def test_nonfinite_optional_output_fails(self):
        predictions = valid_predictions()
        predictions["semantic_logits"][0, 0, 0, 0] = float("nan")
        gate = evaluate_gate(predictions, peak_allocated_mib=1024)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["all_output_tensors_finite"])

    def test_wrong_sequence_dimension_fails(self):
        predictions = valid_predictions()
        predictions["gs_map"] = torch.zeros(1, 3, 2, 12)
        gate = evaluate_gate(predictions, peak_allocated_mib=1024)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["required_sequence_dimension_matches"])

    def test_memory_limit_is_strict(self):
        gate = evaluate_gate(valid_predictions(), peak_allocated_mib=44 * 1024)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["peak_allocated_below_44_gib"])

    def test_summary_reports_only_finite_range(self):
        summary = tensor_summary(torch.tensor([1.0, float("inf"), 3.0]))
        self.assertAlmostEqual(summary["finite_fraction"], 2 / 3)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)

    @mock.patch("scripts.trust4d.probe_dggt.subprocess.check_output")
    def test_source_contract_accepts_locked_clean_tree(self, check_output):
        check_output.side_effect = [EXPECTED_DGGT_COMMIT + "\n", ""]
        result = verify_source_contract(Path("/dggt"))
        self.assertTrue(result["working_tree_clean"])

    @mock.patch("scripts.trust4d.probe_dggt.subprocess.check_output")
    def test_source_contract_rejects_dirty_tree(self, check_output):
        check_output.side_effect = [EXPECTED_DGGT_COMMIT + "\n", " M dggt/model.py\n"]
        with self.assertRaisesRegex(ValueError, "not clean"):
            verify_source_contract(Path("/dggt"))

    @mock.patch("scripts.trust4d.probe_dggt.subprocess.check_output")
    def test_source_contract_rejects_wrong_commit(self, check_output):
        check_output.side_effect = ["deadbeef\n", ""]
        with self.assertRaisesRegex(ValueError, "commit mismatch"):
            verify_source_contract(Path("/dggt"))

    def test_checkpoint_contract_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "model.pt"
            checkpoint.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_checkpoint(checkpoint, expected_size=4)

    def test_selection_manifest_is_rederived(self):
        with tempfile.TemporaryDirectory() as root:
            _, selection_path = self.make_selection(root)
            result = load_verified_selection(selection_path)
        self.assertEqual(result["content"]["count"], 4)

    def test_selection_manifest_rejects_changed_image(self):
        with tempfile.TemporaryDirectory() as root:
            scene, selection_path = self.make_selection(root)
            (scene / "image" / "000000.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_verified_selection(selection_path)


if __name__ == "__main__":
    unittest.main()
