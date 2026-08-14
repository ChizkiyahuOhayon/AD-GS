import unittest

import torch

from scripts.trust4d.probe_dggt import evaluate_gate, tensor_summary


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


if __name__ == "__main__":
    unittest.main()
