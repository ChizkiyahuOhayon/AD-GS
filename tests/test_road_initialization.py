import unittest

import torch

from models.road_initialization import (
    actor_center_samples,
    extract_road_support,
    initialize_road_chart,
)


class RoadInitializationTest(unittest.TestCase):
    def test_actor_centers_are_grouped_by_identity_and_time(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                [1.0, 2.0, 0.0], [3.0, 2.0, 0.0],
                [10.0, 0.0, 0.0], [12.0, 0.0, 0.0],
            ]
        )
        times = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        actor_ids = torch.tensor([3, 3, 3, 3, 8, 8])

        centers = actor_center_samples(xyz, times, actor_ids, min_points=2)

        self.assertTrue(
            torch.equal(centers, torch.tensor([[0.0, 0.0], [1.0, 2.0], [10.0, 0.0]]))
        )

    def test_static_lower_envelope_rejects_elevated_returns(self):
        road_points = []
        elevated_points = []
        for x in range(6):
            for y in range(4):
                road_z = 0.1 * x - 0.05 * y + 0.3
                for offset in (0.1, 0.3, 0.6):
                    road_points.append([x + offset, y + 0.2, road_z])
                    elevated_points.append([x + offset, y + 0.4, road_z + 5.0])
        static_xyz = torch.tensor(road_points + elevated_points)
        query_xy = torch.tensor([[1.5, 1.5], [3.5, 2.5], [4.5, 1.5]])

        support_xy, support_z = extract_road_support(
            static_xyz,
            query_xy,
            search_radius=1.6,
            min_points_per_cell=3,
            min_neighbor_cells=3,
            neighbor_count=9,
        )

        expected = 0.1 * support_xy[:, 0].floor() - 0.05 * support_xy[:, 1].floor() + 0.3
        self.assertEqual(support_xy.shape[0], query_xy.shape[0])
        self.assertTrue(torch.allclose(support_z, expected, atol=0.11, rtol=0.0))

    def test_chart_initialization_preserves_a_planar_trend(self):
        support_xy = torch.tensor(
            [[0.2, 0.3], [1.0, 3.2], [2.7, 1.1], [4.5, 4.2], [5.3, 2.4], [3.1, 5.0]]
        )
        slope = torch.tensor([0.2, -0.1])
        support_z = support_xy @ slope + 1.5
        chart = initialize_road_chart(support_xy, support_z, knot_spacing=2.0)
        query_xy = torch.tensor([[0.5, 0.7], [2.5, 2.1], [4.8, 4.6]])

        height, valid = chart(query_xy)

        self.assertTrue(bool(valid.all()))
        self.assertTrue(
            torch.allclose(height, query_xy @ slope + 1.5, atol=2e-5, rtol=0.0)
        )

    def test_missing_local_support_fails_closed(self):
        static_xyz = torch.tensor(
            [[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0]]
        )

        with self.assertRaisesRegex(ValueError, "no actor query"):
            extract_road_support(
                static_xyz,
                torch.tensor([[100.0, 100.0]]),
                min_points_per_cell=3,
            )


if __name__ == "__main__":
    unittest.main()
