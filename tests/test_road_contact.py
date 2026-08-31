import unittest

import torch

from models.contact_tie import vertical_standard_deviation
from models.road_chart import BicubicRoadChart
from models.road_contact import project_actor_contact_to_chart


class RoadContactTest(unittest.TestCase):
    def test_active_actor_lower_extent_is_tied_to_chart(self):
        xyz = torch.tensor(
            [[2.0, 2.0, 1.0], [4.0, 2.0, 1.4], [6.0, 2.0, 3.0]],
            requires_grad=True,
        )
        scales = torch.full((3, 3), 0.1, requires_grad=True)
        rotations = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 3, requires_grad=True
        )
        actor_ids = torch.tensor([4, 4, 9])
        weights = torch.tensor([0.4, 0.6, 1.0])
        chart = BicubicRoadChart(torch.full((7, 7), 0.5), torch.zeros(2), 2.0)

        projected, diagnostics = project_actor_contact_to_chart(
            xyz,
            scales,
            rotations,
            actor_ids,
            active_actor_ids=torch.tensor([4]),
            sample_weights=weights,
            road_chart=chart,
        )

        support = projected[:, 2] - 2.0 * vertical_standard_deviation(
            scales, rotations
        )
        self.assertAlmostEqual(float(torch.min(support[:2]).detach()), 0.5, places=6)
        self.assertTrue(torch.equal(projected[2], xyz[2]))
        self.assertEqual(diagnostics["actor_count"], 1)
        self.assertEqual(diagnostics["invalid_actor_count"], 0)
        self.assertGreater(float(diagnostics["mean_abs_before"].detach()), 0.0)
        self.assertEqual(float(diagnostics["mean_abs_after"]), 0.0)

        projected.square().sum().backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertTrue(torch.isfinite(rotations.grad).all())

    def test_actor_outside_chart_is_released(self):
        xyz = torch.tensor([[20.0, 20.0, 1.0]])
        scales = torch.full((1, 3), 0.1)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        chart = BicubicRoadChart(torch.zeros((5, 5)), torch.zeros(2), 2.0)

        projected, diagnostics = project_actor_contact_to_chart(
            xyz,
            scales,
            rotations,
            actor_ids=torch.tensor([3]),
            active_actor_ids=torch.tensor([3]),
            sample_weights=torch.ones(1),
            road_chart=chart,
        )

        self.assertTrue(torch.equal(projected, xyz))
        self.assertEqual(diagnostics["actor_count"], 0)
        self.assertEqual(diagnostics["invalid_actor_count"], 1)

    def test_negligible_actor_support_is_released_without_weight_gradients(self):
        xyz = torch.tensor(
            [[2.0, 2.0, 1.0], [3.0, 2.0, 1.2]], requires_grad=True
        )
        scales = torch.full((2, 3), 0.1, requires_grad=True)
        rotations = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2, requires_grad=True
        )
        weights = torch.full((2,), 1e-30, requires_grad=True)
        chart = BicubicRoadChart(torch.zeros((7, 7)), torch.zeros(2))

        projected, diagnostics = project_actor_contact_to_chart(
            xyz,
            scales,
            rotations,
            actor_ids=torch.tensor([3, 3]),
            active_actor_ids=torch.tensor([3]),
            sample_weights=weights,
            road_chart=chart,
        )

        self.assertTrue(torch.equal(projected, xyz))
        self.assertEqual(diagnostics["actor_count"], 0)
        projected.sum().backward()
        self.assertIsNone(weights.grad)
        self.assertTrue(torch.isfinite(xyz.grad).all())


if __name__ == "__main__":
    unittest.main()
