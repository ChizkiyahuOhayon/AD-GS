import unittest

import torch

from models.actor_rigid import (
    canonicalize_actor_points,
    fixed_memberships_from_actor_ids,
    world_points_from_actor_pose,
)
from models.contact_tie import ExtentTiedHeight, canonical_lower_extent


class ActorRigidTest(unittest.TestCase):
    def test_fixed_actor_ids_map_to_stable_sorted_memberships(self):
        actor_ids = torch.tensor([22, 11, 22, 31, 11], dtype=torch.long)
        unique_ids, memberships = fixed_memberships_from_actor_ids(actor_ids)

        self.assertEqual(unique_ids.tolist(), [11, 22, 31])
        self.assertEqual(
            torch.argmax(memberships, dim=1).tolist(), [1, 0, 1, 2, 0]
        )
        self.assertTrue(torch.allclose(memberships.sum(dim=1), torch.ones(5)))

    def test_world_canonical_round_trip_and_gradients(self):
        world_xyz = torch.tensor(
            [[2.0, 3.0, 1.0], [-1.0, 4.0, 0.2]], requires_grad=True
        )
        translations = torch.tensor(
            [[1.0, 2.0, 0.5], [-2.0, 3.0, -0.1]], requires_grad=True
        )
        yaw = torch.tensor([0.7, -1.2], requires_grad=True)

        canonical = canonicalize_actor_points(world_xyz, translations, yaw)
        restored = world_points_from_actor_pose(canonical, translations, yaw)
        self.assertTrue(torch.allclose(restored, world_xyz, atol=1e-6, rtol=0.0))

        canonical.square().sum().backward()
        self.assertIsNotNone(world_xyz.grad)
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(yaw.grad)

    def test_contact_tie_places_world_lower_extent_on_road(self):
        canonical_xyz = torch.tensor(
            [[-0.5, 0.0, 0.2], [0.5, 0.0, 0.4], [0.0, 0.3, 0.3]]
        )
        scales = torch.full((3, 3), 0.1)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
        memberships = torch.ones((3, 1))
        road_height = torch.tensor([1.5])

        tied = ExtentTiedHeight()
        actor_height, _ = tied(
            road_height, canonical_xyz, scales, rotations, memberships
        )
        translations = torch.zeros_like(canonical_xyz)
        translations[:, 2] = actor_height[0]
        yaw = torch.tensor([0.0, 0.5, -0.7])
        world_xyz = world_points_from_actor_pose(canonical_xyz, translations, yaw)

        world_lower_extent = canonical_lower_extent(
            world_xyz, scales, rotations, memberships
        )
        self.assertTrue(
            torch.allclose(world_lower_extent, road_height, atol=1e-6, rtol=0.0)
        )


if __name__ == "__main__":
    unittest.main()
