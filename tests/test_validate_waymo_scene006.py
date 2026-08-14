import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.trust4d.validate_waymo_scene006 import validate_scene


class ValidateWaymoScene006Test(unittest.TestCase):
    def make_scene(self, root, frame_count=86):
        scene = Path(root) / "scene006"
        image_dir = scene / "image"
        image_dir.mkdir(parents=True)
        for index in range(frame_count):
            Image.new("RGB", (2, 1), (index, 0, 0)).save(
                image_dir / f"{index:06d}.jpg"
            )
        is_val = np.zeros(frame_count, dtype=np.bool_)
        is_val[4::4] = True
        np.savez(
            scene / "cameras.npz",
            R=np.repeat(np.eye(3)[None], frame_count, axis=0),
            T=np.zeros((frame_count, 3)),
            K=np.zeros((frame_count, 9)),
            time_stamps=np.arange(frame_count, dtype=np.float32),
            is_val_list=is_val,
        )
        (scene / "points3d.ply").write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float t\n"
            "end_header\n"
            "0 0 0 0\n"
        )
        return scene

    def test_valid_scene_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = validate_scene(self.make_scene(root))
        self.assertTrue(result["passed"])
        self.assertEqual(result["frame_count"], 86)
        self.assertEqual(result["validation_indices"][:2], [4, 8])
        self.assertEqual(result["points3d"]["vertex_count"], 1)

    def test_generic_101_frame_scene_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = validate_scene(self.make_scene(root, 101), 101)
        self.assertTrue(result["passed"])
        self.assertEqual(result["frame_count"], 101)
        self.assertEqual(result["validation_indices"][-1], 100)

    def test_rejects_wrong_split(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root)
            with np.load(scene / "cameras.npz") as metadata:
                values = {key: metadata[key] for key in metadata.files}
            values["is_val_list"][4] = False
            np.savez(scene / "cameras.npz", **values)
            with self.assertRaisesRegex(ValueError, "every-fourth-frame"):
                validate_scene(scene)

    def test_rejects_wrong_camera_shape(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root)
            with np.load(scene / "cameras.npz") as metadata:
                values = {key: metadata[key] for key in metadata.files}
            values["R"] = np.zeros((86, 1))
            np.savez(scene / "cameras.npz", **values)
            with self.assertRaisesRegex(ValueError, "R shape"):
                validate_scene(scene)

    def test_rejects_missing_time_property(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root)
            (scene / "points3d.ply").write_text(
                "ply\n"
                "format ascii 1.0\n"
                "element vertex 1\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
                "0 0 0\n"
            )
            with self.assertRaisesRegex(ValueError, "vertex properties"):
                validate_scene(scene)

    def test_rejects_missing_vertex_payload(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root)
            (scene / "points3d.ply").write_text(
                "ply\n"
                "format ascii 1.0\n"
                "element vertex 1\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float t\n"
                "end_header\n"
            )
            with self.assertRaisesRegex(ValueError, "no vertex payload"):
                validate_scene(scene)


if __name__ == "__main__":
    unittest.main()
