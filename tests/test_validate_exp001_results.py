import json
import tempfile
import unittest
from pathlib import Path

from scripts.trust4d.validate_exp001_results import validate_exp001


def tensor_summary(shape, dtype="torch.float32"):
    numel = 1
    for value in shape:
        numel *= value
    return {
        "shape": shape,
        "dtype": dtype,
        "device": "cuda:0",
        "numel": numel,
        "finite_fraction": 1.0,
        "min": 0.0,
        "max": 1.0,
    }


class ValidateExp001ResultsTest(unittest.TestCase):
    def make_result(self, root):
        result = Path(root) / "exp001"
        result.mkdir()
        selection = {
            "scene": "/data/scene006",
            "count": 4,
            "images": [
                {"index": index, "path": f"/data/{index:06d}.jpg", "sha256": "a" * 64}
                for index in range(4)
            ],
        }
        selection_path = result / "selection.json"
        selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n")
        import hashlib

        selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        outputs = {
            "pose_enc": tensor_summary([1, 4, 9]),
            "world_points": tensor_summary([1, 4, 28, 28, 3]),
            "world_points_conf": tensor_summary([1, 4, 28, 28]),
            "gs_map": tensor_summary([1, 4, 28, 28, 12]),
            "gs_conf": tensor_summary([1, 4, 28, 28]),
            "dynamic_conf": tensor_summary([1, 4, 28, 28, 1]),
            "depth": tensor_summary([1, 4, 28, 28, 1]),
            "depth_conf": tensor_summary([1, 4, 28, 28]),
            "nested": [tensor_summary([1])],
        }
        gate = {
            "missing_required_keys": [],
            "required_outputs_are_tensors": True,
            "required_outputs_nonempty": True,
            "all_output_tensors_finite": True,
            "required_sequence_dimension_matches": True,
            "peak_allocated_not_above_reserved": True,
            "reserved_memory_margin_at_least_4096_mib": True,
            "passed": True,
        }
        metrics = {
            "experiment_id": "EXP-001",
            "source": {
                "commit": "a3276d2bbe4cbb03bcc117830b1836110a27adeb",
                "working_tree_clean": True,
            },
            "checkpoint": {
                "size_bytes": 5_411_266_466,
                "sha256": "b" * 64,
            },
            "runtime_packages": {
                "torch": "2.4.1+cu121",
                "torchvision": "0.19.1+cu121",
                "gsplat": "1.5.3",
                "scikit-learn": "1.5.2",
            },
            "selection_manifest": {
                "content": selection,
                "sha256": selection_sha,
            },
            "environment": {
                "gpu": "NVIDIA A40",
                "cuda_visible_devices": "0",
                "compute_capability": [8, 6],
                "cuda_runtime": "12.1",
                "total_memory_mib": 46068.0,
            },
            "input": tensor_summary([1, 4, 3, 392, 518]),
            "outputs": outputs,
            "model_parameters": 1_000_000,
            "wall_time_seconds": 2.0,
            "peak_memory_allocated_mib": 20_000.0,
            "peak_memory_reserved_mib": 30_000.0,
            "reserved_memory_margin_mib": 16_068.0,
            "gate": gate,
        }
        (result / "metrics.json").write_text(json.dumps(metrics) + "\n")
        (result / "probe_exitcode.txt").write_text("0\n")
        (result / "wall_time_seconds.txt").write_text("2\n")
        for name in (
            "git_remote.txt",
            "git_commit.txt",
            "git_status.txt",
            "nvidia-smi.txt",
            "environment.txt",
            "command.sh",
            "stdout.log",
            "stderr.log",
            "selection.stdout.log",
            "selection.stderr.log",
        ):
            (result / name).write_text("evidence\n")
        return result

    def test_valid_result_passes(self):
        with tempfile.TemporaryDirectory() as root:
            validated = validate_exp001(self.make_result(root))
        self.assertTrue(validated["passed"])
        self.assertEqual(validated["input_shape"], [1, 4, 3, 392, 518])

    def test_nonfinite_output_fails(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.make_result(root)
            metrics_path = result / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["outputs"]["world_points"]["finite_fraction"] = 0.99
            metrics_path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "world_points is not entirely finite"):
                validate_exp001(result)

    def test_low_memory_margin_fails_even_if_probe_claims_pass(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.make_result(root)
            metrics_path = result / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["peak_memory_reserved_mib"] = 43_000.0
            metrics["reserved_memory_margin_mib"] = 3_068.0
            metrics_path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "below 4096"):
                validate_exp001(result)

    def test_changed_selection_fails(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.make_result(root)
            selection_path = result / "selection.json"
            selection = json.loads(selection_path.read_text())
            selection["images"][0]["path"] = "/data/changed.jpg"
            selection_path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "exact selection manifest"):
                validate_exp001(result)

    def test_wrong_gpu_fails(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.make_result(root)
            metrics_path = result / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["environment"]["gpu"] = "NVIDIA RTX 4090"
            metrics_path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "NVIDIA A40"):
                validate_exp001(result)


if __name__ == "__main__":
    unittest.main()
