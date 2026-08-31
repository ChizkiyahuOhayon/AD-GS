import unittest
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

from utils.waymo_camera_protocol import (
    infer_frame_gap,
    is_validation_camera,
    load_camera_ids,
    summarize_camera_metrics,
    validate_camera_split,
)


VERIFY_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "waymo"
    / "verify_cross_camera.py"
)
VERIFY_SPEC = importlib.util.spec_from_file_location("verify_cross_camera", VERIFY_PATH)
VERIFY_MODULE = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(VERIFY_MODULE)


class WaymoCameraProtocolTest(unittest.TestCase):
    def test_default_split_preserves_all_selected_training_cameras(self):
        train = validate_camera_split([0, 1, 2])

        self.assertEqual(train, (0, 1, 2))
        self.assertFalse(is_validation_camera(False, 2, train))
        self.assertTrue(is_validation_camera(True, 2, train))

    def test_front_only_training_keeps_side_cameras_out_of_training(self):
        train = validate_camera_split([0, 1, 2], [0])

        self.assertFalse(is_validation_camera(False, 0, train))
        self.assertTrue(is_validation_camera(False, 1, train))
        self.assertTrue(is_validation_camera(False, 2, train))
        self.assertTrue(is_validation_camera(True, 0, train))

    def test_training_cameras_must_be_selected(self):
        with self.assertRaisesRegex(ValueError, "not selected"):
            validate_camera_split([0, 1], [2])

    def test_explicit_camera_ids_override_legacy_modulo_fallback(self):
        explicit = load_camera_ids(
            {"camera_ids": np.asarray([0, 2, 0, 2])},
            image_count=4,
            fallback_num_cameras=1,
        )
        legacy = load_camera_ids({}, image_count=4, fallback_num_cameras=2)

        self.assertEqual(explicit.tolist(), [0, 2, 0, 2])
        self.assertEqual(legacy.tolist(), [0, 1, 0, 1])

    def test_frame_gap_depends_on_frames_not_camera_count(self):
        time_stamps = np.repeat(np.arange(5), 3)

        self.assertEqual(infer_frame_gap(time_stamps), 0.2)

    def test_metrics_are_aggregated_without_mixing_cameras(self):
        summary = summarize_camera_metrics(
            {
                2: [{"PSNR": 10.0}, {"PSNR": 14.0}],
                0: [{"PSNR": 30.0}],
            }
        )

        self.assertEqual(list(summary), ["0", "2"])
        self.assertEqual(summary["0"], {"count": 1, "PSNR": 30.0})
        self.assertEqual(summary["2"], {"count": 2, "PSNR": 12.0})

    def test_cross_camera_audit_rejects_side_camera_training_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            images = scene / "image"
            images.mkdir()
            camera_ids = np.tile(np.asarray([0, 1, 2]), 4)
            is_validation = np.tile(np.asarray([False, True, True]), 4)
            is_validation[9] = True
            for index in range(camera_ids.size):
                (images / "{:06d}.jpg".format(index)).write_bytes(b"image")
            np.savez(
                scene / "cameras.npz",
                camera_ids=camera_ids,
                is_val_list=is_validation,
                time_stamps=np.repeat(np.arange(4), 3),
            )

            result = VERIFY_MODULE.audit_cross_camera_scene(scene)
            self.assertTrue(result["passed"])
            is_validation[1] = False
            np.savez(
                scene / "cameras.npz",
                camera_ids=camera_ids,
                is_val_list=is_validation,
                time_stamps=np.repeat(np.arange(4), 3),
            )
            with self.assertRaisesRegex(ValueError, "only FRONT"):
                VERIFY_MODULE.audit_cross_camera_scene(scene)


if __name__ == "__main__":
    unittest.main()
