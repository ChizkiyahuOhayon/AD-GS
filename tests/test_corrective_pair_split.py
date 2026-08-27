from argparse import ArgumentParser

import torch

from arguments import OptimizationParams
from utils.densification_utils import corrective_pair_samples


def test_corrective_pair_split_is_opt_in():
    parser = ArgumentParser()
    OptimizationParams(parser, None)

    assert parser.parse_args([]).corrective_pair_split is False
    assert parser.parse_args(["--corrective_pair_split"]).corrective_pair_split is True


def test_corrective_pair_uses_one_mahalanobis_step():
    scales = torch.tensor([[2.0, 1.0, 0.5]])
    rotations = torch.eye(3).unsqueeze(0)
    exp_avg = torch.tensor([[1.0, 0.0, 0.0]])
    exp_avg_sq = torch.ones_like(exp_avg)
    fallback = torch.full((2, 3), 7.0)

    samples = corrective_pair_samples(
        scales, rotations, exp_avg, exp_avg_sq, fallback, 1e-15
    )

    torch.testing.assert_close(samples[0], torch.zeros(3))
    torch.testing.assert_close(samples[1], torch.tensor([-2.0, 0.0, 0.0]))
    torch.testing.assert_close(torch.linalg.vector_norm(samples[1] / scales[0]), torch.tensor(1.0))


def test_corrective_pair_follows_world_descent_after_rotation():
    rotations = torch.tensor([[[0.0, -1.0, 0.0],
                               [1.0,  0.0, 0.0],
                               [0.0,  0.0, 1.0]]])
    scales = torch.tensor([[2.0, 1.0, 0.5]])
    exp_avg = torch.tensor([[1.0, 0.0, 0.0]])
    exp_avg_sq = torch.ones_like(exp_avg)
    fallback = torch.full((2, 3), 7.0)

    samples = corrective_pair_samples(
        scales, rotations, exp_avg, exp_avg_sq, fallback, 1e-15
    )
    world_offset = torch.bmm(rotations, samples[1:].unsqueeze(-1)).squeeze()
    descent = -exp_avg[0]

    assert torch.dot(world_offset, descent) > 0
    torch.testing.assert_close(torch.linalg.vector_norm(samples[1] / scales[0]), torch.tensor(1.0))


def test_corrective_pair_falls_back_for_zero_direction():
    scales = torch.ones((1, 3))
    rotations = torch.eye(3).unsqueeze(0)
    exp_avg = torch.zeros((1, 3))
    exp_avg_sq = torch.zeros((1, 3))
    fallback = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    samples = corrective_pair_samples(
        scales, rotations, exp_avg, exp_avg_sq, fallback, 1e-15
    )

    torch.testing.assert_close(samples, fallback)


def test_corrective_pair_falls_back_per_parent():
    scales = torch.ones((2, 3))
    rotations = torch.eye(3).repeat(2, 1, 1)
    exp_avg = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    exp_avg_sq = torch.ones_like(exp_avg)
    fallback = torch.tensor([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
    ])

    samples = corrective_pair_samples(
        scales, rotations, exp_avg, exp_avg_sq, fallback, 1e-15
    )

    torch.testing.assert_close(samples[0], torch.zeros(3))
    torch.testing.assert_close(samples[2], torch.tensor([-1.0, 0.0, 0.0]))
    torch.testing.assert_close(samples[1], fallback[1])
    torch.testing.assert_close(samples[3], fallback[3])
