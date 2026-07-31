import unittest

import numpy as np

from models.geometry_metrics import (
    alpha_composited_depth,
    paired_sparse_depth_metrics,
    select_sparse_actor_depth,
)


class GeometryMetricsTest(unittest.TestCase):
    def test_sparse_selection_zbuffers_before_actor_filter(self):
        uv = np.array([[1.2, 1.1], [1.4, 1.3], [2.0, 1.0], [5.0, 5.0]])
        depth = np.array([8.0, 3.0, 4.0, 1.0])
        actor_map = np.zeros((3, 4), dtype=np.int64)
        actor_map[1, 1] = 11
        actor_map[1, 2] = 22

        samples = select_sparse_actor_depth(uv, depth, actor_map, [11])

        self.assertTrue(np.array_equal(samples["u"], np.array([1])))
        self.assertTrue(np.array_equal(samples["v"], np.array([1])))
        self.assertTrue(np.array_equal(samples["depth"], np.array([3.0])))
        self.assertTrue(np.array_equal(samples["actor_id"], np.array([11])))

    def test_alpha_composited_depth_normalizes_inverse_depth(self):
        inverse_depth = np.array([0.25, 0.10])
        opacity = np.array([0.5, 0.5])
        predicted = alpha_composited_depth(inverse_depth, opacity)
        self.assertTrue(np.allclose(predicted, np.array([2.0, 5.0])))

    def test_paired_metrics_use_one_common_pixel_and_actor_set(self):
        gt = np.array([2.0, 4.0, 5.0, 8.0])
        actor_ids = np.array([11, 11, 22, 33])
        baseline_depth = np.array([2.2, 3.2, 6.0, 8.0])
        oracle_depth = np.array([2.0, 3.6, 5.5, 8.0])
        baseline_opacity = np.array([0.5, 0.5, 0.5, 0.05])
        oracle_opacity = np.array([0.5, 0.5, 0.05, 0.5])

        result = paired_sparse_depth_metrics(
            gt,
            actor_ids,
            baseline_opacity / baseline_depth,
            baseline_opacity,
            oracle_opacity / oracle_depth,
            oracle_opacity,
            min_opacity=0.1,
            min_pixels=2,
            min_actors=1,
        )

        self.assertEqual(result["pixel_count"], 2)
        self.assertEqual(result["actor_ids"], [11])
        self.assertAlmostEqual(result["baseline_absrel"], 0.15)
        self.assertAlmostEqual(result["oracle_absrel"], 0.05)
        self.assertAlmostEqual(result["relative_improvement"], 2.0 / 3.0)

    def test_paired_metrics_reject_insufficient_common_support(self):
        with self.assertRaisesRegex(ValueError, "common pixels"):
            paired_sparse_depth_metrics(
                np.array([2.0]),
                np.array([11]),
                np.array([0.25]),
                np.array([0.5]),
                np.array([0.25]),
                np.array([0.5]),
                min_pixels=2,
                min_actors=1,
            )


if __name__ == "__main__":
    unittest.main()
