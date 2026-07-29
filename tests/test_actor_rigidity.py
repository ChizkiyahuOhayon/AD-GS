import unittest

import torch

from analysis.actor_rigidity import fit_actor_rigid_trajectories, summarize_fit


def apply_poses(canonical_xyz, actor_ids, translations, yaw):
    frames = []
    unique_ids = torch.unique(actor_ids, sorted=True)
    for time_index in range(translations.shape[0]):
        world = torch.empty_like(canonical_xyz)
        for actor_index, actor_id in enumerate(unique_ids):
            mask = actor_ids == actor_id
            points = canonical_xyz[mask]
            cosine = torch.cos(yaw[time_index, actor_index])
            sine = torch.sin(yaw[time_index, actor_index])
            world[mask, 0] = cosine * points[:, 0] - sine * points[:, 1]
            world[mask, 1] = sine * points[:, 0] + cosine * points[:, 1]
            world[mask, 2] = points[:, 2]
            world[mask] += translations[time_index, actor_index]
        frames.append(world)
    return torch.stack(frames)


class ActorRigidityTest(unittest.TestCase):
    def setUp(self):
        self.actor_ids = torch.tensor([11, 11, 11, 11, 22, 22, 22, 22])
        self.canonical_xyz = torch.tensor(
            [
                [-1.0, -0.5, 0.2],
                [1.0, -0.5, 0.2],
                [1.0, 0.5, 0.8],
                [-1.0, 0.5, 0.8],
                [-0.4, -0.4, 0.1],
                [0.6, -0.4, 0.1],
                [0.6, 0.4, 0.5],
                [-0.4, 0.4, 0.5],
            ]
        )
        self.translations = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.2, 0.1], [-0.5, 0.4, -0.2]],
                [[2.0, 0.7, 0.2], [-1.0, 1.0, -0.3]],
            ]
        )
        self.yaw = torch.tensor([[0.0, 0.0], [0.3, -0.4], [0.7, -0.8]])
        self.world_xyz = apply_poses(
            self.canonical_xyz, self.actor_ids, self.translations, self.yaw
        )

    def test_exact_multi_actor_trajectories_recover_pose(self):
        fit = fit_actor_rigid_trajectories(self.world_xyz, self.actor_ids)

        self.assertEqual(fit["actor_ids"].tolist(), [11, 22])
        self.assertEqual(fit["reference_indices"].tolist(), [0, 0])
        self.assertTrue(torch.all(fit["valid"]))
        self.assertTrue(
            torch.allclose(fit["translations"], self.translations, atol=1e-6)
        )
        self.assertTrue(torch.allclose(fit["yaw"], self.yaw, atol=1e-6))
        self.assertLess(float(torch.max(fit["rmse_3d"])), 1e-6)

    def test_nonrigid_deformation_is_actor_local_and_detectable(self):
        nonrigid = self.world_xyz.clone()
        nonrigid[2, 0, 0] += 0.8
        fit = fit_actor_rigid_trajectories(nonrigid, self.actor_ids)

        self.assertGreater(float(fit["rmse_3d"][2, 0]), 0.1)
        self.assertLess(float(torch.max(fit["rmse_3d"][:, 1])), 1e-6)

    def test_insufficient_weighted_support_is_invalid(self):
        weights = torch.ones(self.world_xyz.shape[:2])
        weights[1, :4] = 0.0
        weights[1, 0] = 1.0
        fit = fit_actor_rigid_trajectories(
            self.world_xyz, self.actor_ids, weights=weights, min_points=3
        )

        self.assertFalse(bool(fit["valid"][1, 0]))
        self.assertTrue(bool(torch.isnan(fit["rmse_3d"][1, 0])))
        self.assertTrue(bool(fit["valid"][1, 1]))

    def test_degenerate_horizontal_support_is_reported_without_fake_metric(self):
        world_xyz = torch.zeros((2, 3, 3), dtype=torch.float32)
        world_xyz[:, :, 2] = torch.tensor([0.0, 1.0, 2.0])
        actor_ids = torch.tensor([11, 11, 11])

        fit = fit_actor_rigid_trajectories(world_xyz, actor_ids)
        summary = summarize_fit(fit)[0]

        self.assertFalse(bool(fit["valid"].any()))
        self.assertEqual(summary["valid_fraction"], 0.0)
        self.assertIsNone(summary["rmse_3d"]["median"])
        self.assertIsNone(summary["rmse_3d"]["max"])


if __name__ == "__main__":
    unittest.main()
