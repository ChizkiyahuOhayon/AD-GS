import json
import tempfile
import unittest
from pathlib import Path

from scripts.trust4d.validate_base001_results import validate_run


class ValidateBase001ResultsTest(unittest.TestCase):
    def make_run(self, root, test_overrides=None):
        run = Path(root) / "run"
        checkpoint = run / "point_cloud/iteration_60000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "point_cloud.ply").write_bytes(b"ply\ncheckpoint")
        (checkpoint / "env.pth").write_bytes(b"weights")
        (run / "cfg_args").write_text("Namespace(iterations=60000)\n")
        metrics = {
            "PSNR": 34.9363,
            "SSIM": 0.95216,
            "LPIPS(VGG)": 0.18436,
            "LPIPS(ALEX)": 0.12,
            "FPS": 30.0,
        }
        train_metrics = dict(metrics, PSNR=38.0)
        metrics.update(test_overrides or {})
        (run / "results.json").write_text(json.dumps({"ours_60000": metrics}))
        (run / "results-train.json").write_text(
            json.dumps({"ours_60000": train_metrics})
        )
        return run

    def test_matching_run_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = validate_run(self.make_run(root))
        self.assertTrue(result["passed"])
        self.assertEqual(result["experiment_id"], "BASE-001")
        self.assertEqual(result["absolute_deviations"]["PSNR"], 0.0)

    def test_boundary_tolerance_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = validate_run(self.make_run(root, {"PSNR": 34.4363}))
        self.assertTrue(result["passed"])

    def test_metric_outside_tolerance_fails(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root, {"LPIPS(VGG)": 0.205})
            with self.assertRaisesRegex(ValueError, "tolerance failed"):
                validate_run(run)

    def test_nonfinite_metric_fails(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root, {"PSNR": float("nan")})
            with self.assertRaisesRegex(ValueError, "not finite"):
                validate_run(run)

    def test_missing_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root)
            (run / "point_cloud/iteration_60000/env.pth").unlink()
            with self.assertRaisesRegex(ValueError, "environment_map"):
                validate_run(run)


if __name__ == "__main__":
    unittest.main()
