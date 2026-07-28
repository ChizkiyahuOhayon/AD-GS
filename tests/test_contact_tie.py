import math
import unittest

import torch

from models.contact_tie import (
    ExtentTiedHeight,
    FreeOffsetHeight,
    canonical_lower_extent,
    vertical_standard_deviation,
)


class ContactTieTest(unittest.TestCase):
    def test_vertical_standard_deviation_respects_rotation(self):
        scales = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        rotations = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0],
            ]
        )

        actual = vertical_standard_deviation(scales, rotations)
        expected = torch.tensor([3.0, 1.0])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))

    def test_fixed_membership_lower_extent(self):
        canonical_xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 10.0], [0.0, 0.0, 11.0]]
        )
        scales = torch.full((4, 3), 0.1)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
        memberships = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )

        actual = canonical_lower_extent(
            canonical_xyz, scales, rotations, memberships, quantile=0.05
        )
        expected = torch.tensor([-0.2, 9.8])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))

    def test_extent_tie_has_no_independent_height_and_propagates_gradient(self):
        canonical_xyz = torch.tensor(
            [[0.0, 0.0, 0.4], [0.0, 0.0, 0.8]], requires_grad=True
        )
        scales = torch.full((2, 3), 0.1, requires_grad=True)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        memberships = torch.ones((2, 1))
        road_height = torch.tensor([0.2], requires_grad=True)
        model = ExtentTiedHeight()

        actor_height, lower_extent = model(
            road_height, canonical_xyz, scales, rotations, memberships
        )
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)
        self.assertTrue(torch.allclose(actor_height + lower_extent, road_height))

        actor_height.sum().backward()
        self.assertIsNotNone(canonical_xyz.grad)
        self.assertIsNotNone(scales.grad)
        self.assertTrue(torch.allclose(road_height.grad, torch.ones_like(road_height)))

    def test_extent_scales_with_canonical_gauge(self):
        canonical_xyz = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.7]])
        scales = torch.full((2, 3), 0.1)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        memberships = torch.ones((2, 1))

        extent = canonical_lower_extent(canonical_xyz, scales, rotations, memberships)
        gauge_scale = 2.5
        scaled_extent = canonical_lower_extent(
            gauge_scale * canonical_xyz,
            gauge_scale * scales,
            rotations,
            memberships,
        )
        self.assertTrue(
            torch.allclose(scaled_extent, gauge_scale * extent, atol=1e-6, rtol=0.0)
        )

    def test_free_offset_can_absorb_an_arbitrary_height_shift(self):
        model = FreeOffsetHeight(torch.tensor([0.4]))
        road_height = torch.tensor([0.0])
        target_height = torch.tensor([1.3])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        optimizer.zero_grad()
        loss = torch.sum((model(road_height) - target_height) ** 2)
        loss.backward()
        optimizer.step()

        self.assertTrue(
            torch.allclose(model(road_height), target_height, atol=1e-6, rtol=0.0)
        )


if __name__ == "__main__":
    unittest.main()
