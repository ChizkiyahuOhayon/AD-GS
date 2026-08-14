import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.trust4d.build_exp002_query_manifest import (
    ANCHORS,
    binary_erode_full_neighborhood,
    build_query_manifest,
    dggt_crop_geometry,
    intervention_indices,
    select_grid_queries,
    transform_object_mask,
)


FRAME_COUNTS = {"scene006": 86, "scene026": 101, "scene090": 103}


def make_scene(root, scene_name, frame_count):
    scene = Path(root) / scene_name
    image_dir = scene / "image"
    semantic_dir = scene / "semantic"
    image_dir.mkdir(parents=True)
    semantic_dir.mkdir()
    for index in range(frame_count):
        Image.new("RGB", (64, 64), (index % 255, 0, 0)).save(
            image_dir / f"{index:06d}.jpg"
        )
    for anchor in ANCHORS:
        np.save(
            semantic_dir / f"mask_{anchor:06d}.npy",
            np.ones((64, 64), dtype=np.uint16),
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
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float t\nend_header\n0 0 0 0\n"
    )
    return scene


class BuildExp002QueryManifestTest(unittest.TestCase):
    def test_dggt_geometry_matches_released_crop_math(self):
        geometry = dggt_crop_geometry(1920, 1280)
        self.assertEqual(geometry["resized_width"], 518)
        self.assertEqual(geometry["resized_height"], 350)
        self.assertEqual(geometry["output_height"], 350)
        self.assertEqual(geometry["crop_top"], 0)

        tall = dggt_crop_geometry(600, 1200)
        self.assertGreater(tall["resized_height"], 518)
        self.assertEqual(tall["output_height"], 518)
        self.assertEqual(
            tall["crop_top"], (tall["resized_height"] - 518) // 2
        )

    def test_five_pixel_erosion_uses_false_padding(self):
        mask = np.ones((20, 20), dtype=np.bool_)
        eroded = binary_erode_full_neighborhood(mask, iterations=5)
        self.assertTrue(eroded[5, 5])
        self.assertTrue(eroded[14, 14])
        self.assertFalse(eroded[4, 5])
        self.assertEqual(int(eroded.sum()), 100)

    def test_grid_uses_cell_centers_and_deterministic_subsampling(self):
        mask = np.ones((518, 518), dtype=np.bool_)
        candidates, selected, ranks = select_grid_queries(mask)
        self.assertEqual(candidates[0], [8, 8])
        self.assertEqual(candidates[1], [24, 8])
        self.assertEqual(candidates[-1], [504, 504])
        self.assertEqual(len(selected), 128)
        self.assertEqual(ranks[0], 0)
        self.assertEqual(ranks[-1], len(candidates) - 1)
        self.assertEqual(len(set(ranks)), 128)

    def test_mask_transform_uses_nearest_and_exact_output_shape(self):
        geometry = dggt_crop_geometry(64, 64)
        mask = np.zeros((64, 64), dtype=np.bool_)
        mask[16:48, 16:48] = True
        transformed = transform_object_mask(mask, geometry)
        self.assertEqual(transformed.shape, (518, 518))
        self.assertTrue(transformed[259, 259])
        self.assertFalse(transformed[0, 0])

    def test_interventions_are_exact(self):
        self.assertEqual(
            intervention_indices(5),
            {
                "original": [5, 10, 15, 19],
                "reverse": [19, 15, 10, 5],
                "sparse": [5, 15, 19],
                "interior_shifted": [5, 9, 14, 19],
            },
        )

    def test_three_scene_manifest_has_twelve_train_only_windows(self):
        with tempfile.TemporaryDirectory() as root:
            scenes = [
                make_scene(root, name, count) for name, count in FRAME_COUNTS.items()
            ]
            manifest = build_query_manifest(scenes)
        self.assertEqual(manifest["total_preselected_windows"], 12)
        self.assertEqual(manifest["query_supported_windows"], 12)
        self.assertEqual([item["scene"] for item in manifest["scenes"]], list(FRAME_COUNTS))
        for scene in manifest["scenes"]:
            for window in scene["windows"]:
                self.assertEqual(window["selected_query_count"], 128)
                self.assertTrue(window["valid_query_support"])
                for record in window["referenced_images"]:
                    self.assertNotEqual(record["index"] % 4, 0)


if __name__ == "__main__":
    unittest.main()
