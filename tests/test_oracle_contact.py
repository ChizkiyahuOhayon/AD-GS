import unittest

import torch

from models.oracle_contact import (
    build_oracle_contact_tracks,
    interpolate_track_height,
    project_actor_contact,
)


class OracleContactTest(unittest.TestCase):
    def test_track_builder_keeps_smooth_actor_and_rejects_identity_jump(self):
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.2], [0.2, 0.0, 0.4],
                [0.1, 0.0, 0.3], [0.3, 0.0, 0.5],
                [0.2, 0.0, 0.4], [0.4, 0.0, 0.6],
                [0.0, 0.0, 1.0], [0.2, 0.0, 1.2],
                [20.0, 0.0, 1.1], [20.2, 0.0, 1.3],
                [40.0, 0.0, 1.2], [40.2, 0.0, 1.4],
            ]
        )
        times = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]).float()
        actor_ids = torch.tensor([11] * 6 + [22] * 6)

        tracks = build_oracle_contact_tracks(
            xyz,
            times,
            actor_ids,
            min_points_per_frame=2,
            min_frames=3,
            max_centroid_jump=5.0,
            quantile=0.5,
        )

        self.assertEqual(list(tracks), [11])
        self.assertTrue(torch.equal(tracks[11]["times"], torch.tensor([0.0, 1.0, 2.0])))
        self.assertTrue(torch.allclose(tracks[11]["heights"], torch.tensor([0.2, 0.3, 0.4])))

    def test_track_height_interpolates_and_clamps(self):
        times = torch.tensor([0.0, 0.5, 1.0])
        heights = torch.tensor([1.0, 2.0, 4.0])

        self.assertAlmostEqual(float(interpolate_track_height(-1.0, times, heights)), 1.0)
        self.assertAlmostEqual(float(interpolate_track_height(0.75, times, heights)), 3.0)
        self.assertAlmostEqual(float(interpolate_track_height(2.0, times, heights)), 4.0)

    def test_projection_aligns_lower_support_and_preserves_gradients(self):
        xyz = torch.tensor(
            [[0.0, 0.0, 2.0], [1.0, 0.0, 3.0], [2.0, 0.0, 4.0], [5.0, 0.0, 7.0]],
            requires_grad=True,
        )
        scales = torch.full((4, 3), 0.1, requires_grad=True)
        rotations = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 4,
            requires_grad=True,
        )
        actor_ids = torch.tensor([11, 11, 11, 99])
        tracks = {
            11: {
                "times": torch.tensor([0.0, 1.0]),
                "heights": torch.tensor([0.5, 1.5]),
            }
        }

        projected, diagnostics = project_actor_contact(
            xyz,
            scales,
            rotations,
            actor_ids,
            query_time=0.5,
            tracks=tracks,
            quantile=0.5,
            sigma_multiplier=2.0,
        )

        support = projected[:3, 2] - 0.2
        self.assertAlmostEqual(float(torch.quantile(support, 0.5).detach()), 1.0, places=6)
        self.assertTrue(torch.equal(projected[:, :2], xyz[:, :2]))
        self.assertAlmostEqual(float(projected[3, 2].detach()), 7.0)
        self.assertEqual(diagnostics["actor_count"], 1)
        self.assertGreater(float(diagnostics["mean_abs_before"].detach()), 0.0)
        self.assertLess(float(diagnostics["mean_abs_after"].detach()), 1e-6)

        projected.square().sum().backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertTrue(torch.isfinite(rotations.grad).all())


if __name__ == "__main__":
    unittest.main()
