import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.trust4d.select_waymo_training_frames import select_training_frames


class SelectWaymoTrainingFramesTest(unittest.TestCase):
    def make_scene(self, root, flags, time_stamps=None):
        scene = Path(root) / "scene006"
        image_dir = scene / "image"
        image_dir.mkdir(parents=True)
        for index in range(len(flags)):
            (image_dir / f"{index:06d}.jpg").write_bytes(bytes([index]))
        if time_stamps is None:
            time_stamps = np.arange(len(flags), dtype=np.float32)
        np.savez(
            scene / "cameras.npz",
            is_val_list=np.asarray(flags, dtype=np.bool_),
            time_stamps=np.asarray(time_stamps),
        )
        return scene

    def test_selects_first_four_training_frames_in_chronological_order(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root, [False, False, True, False, False])
            result = select_training_frames(scene, count=4)
        self.assertEqual([item["index"] for item in result["images"]], [0, 1, 3, 4])
        self.assertEqual(
            [item["time_stamp"] for item in result["images"]], [0.0, 1.0, 3.0, 4.0]
        )

    def test_rejects_image_metadata_length_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root, [False, False])
            (scene / "image" / "000002.jpg").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "lengths must match"):
                select_training_frames(scene, count=1)

    def test_rejects_insufficient_training_frames(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root, [False, True, True])
            with self.assertRaisesRegex(ValueError, "requested 2 training frames"):
                select_training_frames(scene, count=2)


if __name__ == "__main__":
    unittest.main()
