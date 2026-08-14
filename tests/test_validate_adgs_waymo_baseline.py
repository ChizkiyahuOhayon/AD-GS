import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.trust4d.validate_adgs_waymo_baseline import (
    read_ply_vertices,
    validate_baseline_scene,
)


class ValidateAdgsWaymoBaselineTest(unittest.TestCase):
    def make_scene(self, root):
        scene = Path(root) / "scene006"
        for directory in ("image", "depth", "semantic", "sky", "flow"):
            (scene / directory).mkdir(parents=True, exist_ok=True)

        is_val = np.zeros(86, dtype=np.bool_)
        is_val[4::4] = True
        for index in range(86):
            Image.new("RGB", (2, 1), (index, 0, 0)).save(
                scene / "image" / f"{index:06d}.jpg"
            )
            np.save(scene / "depth" / f"{index:06d}.npy", np.array([[[0.0], [1.0]]]))
            semantic = np.zeros((1, 2), dtype=np.uint16)
            sky = np.zeros((1, 2), dtype=np.uint16)
            if index == 0:
                semantic[0, 0] = 1
                sky[0, 1] = 1
            np.save(scene / "semantic" / f"mask_{index:06d}.npy", semantic)
            np.save(scene / "sky" / f"mask_{index:06d}.npy", sky)

        np.savez(
            scene / "cameras.npz",
            R=np.repeat(np.eye(3)[None], 86, axis=0),
            T=np.zeros((86, 3)),
            K=np.zeros((86, 9)),
            time_stamps=np.arange(86, dtype=np.float32),
            is_val_list=is_val,
        )
        (scene / "points3d.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 2\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float t\nproperty float obj\nend_header\n"
            "0 0 0 0 0\n1 0 0 0 1\n"
        )
        (scene / "colmap.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"
        )

        record = np.empty(6, dtype=object)
        record[:] = [
            1.0,
            np.eye(3),
            np.eye(3),
            np.zeros(3),
            np.zeros((2, 1, 2)),
            np.ones((1, 2)),
        ]
        records = np.empty((1, 6), dtype=object)
        records[0] = record
        np.savez(scene / "flow" / "000000.npz", flow=records)
        return scene

    def test_complete_scene_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = validate_baseline_scene(self.make_scene(root))
        self.assertTrue(result["passed"])
        self.assertEqual(result["points3d"]["object_vertex_count"], 1)
        self.assertEqual(result["required_dynamic_training_flow_count"], 1)

    def test_missing_dynamic_flow_fails(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root)
            (scene / "flow" / "000000.npz").unlink()
            with self.assertRaisesRegex(ValueError, "missing flow"):
                validate_baseline_scene(scene)

    def test_constant_depth_fails(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root)
            np.save(scene / "depth" / "000000.npy", np.zeros((1, 2, 1)))
            with self.assertRaisesRegex(ValueError, "nonconstant"):
                validate_baseline_scene(scene)

    def test_unsegmented_points_fail(self):
        with tempfile.TemporaryDirectory() as root:
            scene = self.make_scene(root)
            (scene / "points3d.ply").write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float t\nend_header\n0 0 0 0\n"
            )
            with self.assertRaisesRegex(ValueError, "obj"):
                validate_baseline_scene(scene)

    def test_reads_binary_little_endian_vertices(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "points.ply"
            header = (
                "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float t\nproperty float obj\nend_header\n"
            ).encode("ascii")
            path.write_bytes(
                header + struct.pack("<ffffffffff", 0, 0, 0, 0, 0, 1, 0, 0, 1, 1)
            )
            summary, vertices = read_ply_vertices(path, {"x", "y", "z", "t", "obj"})
            objects = np.asarray(vertices["obj"]).tolist()
        self.assertEqual(summary["vertex_count"], 2)
        self.assertEqual(objects, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
