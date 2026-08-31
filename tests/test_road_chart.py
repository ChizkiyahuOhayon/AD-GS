import unittest

import torch

from models.actor_rigid import world_points_from_actor_pose
from models.contact_tie import ExtentTiedHeight, canonical_lower_extent
from models.road_chart import BicubicRoadChart


def planar_controls(height, width, origin, spacing, slope, intercept):
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    world_x = origin[0] + spacing * x
    world_y = origin[1] + spacing * y
    return slope[0] * world_x + slope[1] * world_y + intercept


class RoadChartTest(unittest.TestCase):
    def test_cardinal_cubic_chart_reproduces_a_plane(self):
        origin = torch.tensor([10.0, -4.0])
        spacing = 2.0
        slope = torch.tensor([0.3, -0.2])
        controls = planar_controls(7, 8, origin, spacing, slope, 1.7)
        chart = BicubicRoadChart(controls, origin, spacing)
        xy = torch.tensor([[12.4, -1.0], [17.5, 2.2], [20.1, 3.6]])

        height, valid = chart(xy)
        expected = xy @ slope + 1.7

        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(height, expected, atol=2e-6, rtol=0.0))

    def test_chart_propagates_control_and_position_gradients(self):
        origin = torch.tensor([0.0, 0.0])
        slope = torch.tensor([0.4, -0.3])
        chart = BicubicRoadChart(
            planar_controls(6, 6, origin, 2.0, slope, 0.5),
            origin,
        )
        xy = torch.tensor([[3.2, 4.6]], requires_grad=True)

        height, valid = chart(xy)
        self.assertTrue(bool(valid.item()))
        height.backward()

        self.assertIsNotNone(chart.control_heights.grad)
        self.assertTrue(torch.allclose(xy.grad[0], slope, atol=1e-6, rtol=0.0))

    def test_out_of_support_queries_are_explicitly_invalid(self):
        chart = BicubicRoadChart(torch.zeros((5, 5)), torch.zeros(2))

        height, valid = chart(torch.tensor([[0.0, 0.0], [4.0, 4.0]]))

        self.assertEqual(valid.tolist(), [False, True])
        self.assertTrue(torch.isfinite(height).all())

    def test_large_grid_out_of_support_queries_do_not_overflow_indices(self):
        chart = BicubicRoadChart(torch.zeros((10, 68)), torch.zeros(2))

        height, valid = chart(torch.tensor([[1e6, 1e6]]))

        self.assertEqual(valid.tolist(), [False])
        self.assertTrue(torch.isfinite(height).all())

    def test_state_dict_preserves_world_chart_definition(self):
        chart = BicubicRoadChart(torch.ones((5, 6)), torch.tensor([3.0, -2.0]), 1.5)
        restored = BicubicRoadChart(torch.zeros((5, 6)), torch.zeros(2), 2.0)

        restored.load_state_dict(chart.state_dict())

        self.assertTrue(torch.equal(restored.control_heights, chart.control_heights))
        self.assertTrue(torch.equal(restored.origin_xy, chart.origin_xy))
        self.assertTrue(torch.equal(restored.knot_spacing, chart.knot_spacing))

    def test_chart_height_composes_with_extent_tie(self):
        origin = torch.zeros(2)
        chart = BicubicRoadChart(
            planar_controls(6, 6, origin, 2.0, torch.tensor([0.1, 0.2]), 0.0),
            origin,
        )
        actor_xy = torch.tensor([[3.0, 3.0], [5.0, 4.0]])
        road_height, valid = chart(actor_xy)
        canonical_xyz = torch.tensor(
            [[-0.2, 0.0, 0.3], [0.2, 0.0, 0.4], [-0.3, 0.0, 0.5], [0.3, 0.0, 0.6]]
        )
        scales = torch.full((4, 3), 0.1)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
        memberships = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )

        actor_height, _ = ExtentTiedHeight()(
            road_height, canonical_xyz, scales, rotations, memberships
        )
        translations = memberships @ torch.cat((actor_xy, actor_height[:, None]), dim=1)
        world_xyz = world_points_from_actor_pose(
            canonical_xyz,
            translations,
            torch.zeros(4),
        )
        lower_extent = canonical_lower_extent(
            world_xyz, scales, rotations, memberships
        )

        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(lower_extent, road_height, atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
