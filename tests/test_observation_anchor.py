import math
import unittest

import torch

from utils.func_utils import get_func_result, get_pointwise_func_result
from utils.general_utils import quaternion_conjugate, quaternion_multiply


class PointwiseTemporalFunctionTest(unittest.TestCase):
    def test_linear_bspline_uses_each_row_time(self):
        times = torch.tensor([[0.25], [0.50], [1.00]])
        params = torch.tensor(
            [
                [[0.0, 2.0, 4.0]],
                [[1.0, 3.0, 5.0]],
                [[2.0, 4.0, 6.0]],
            ],
            requires_grad=True,
        )

        values = get_pointwise_func_result(times, params, [3, 1, 0, 0, 0, 0])

        torch.testing.assert_close(values[:, 0], torch.tensor([1.0, 3.0, 6.0]))
        values.sum().backward()
        self.assertIsNotNone(params.grad)

    def test_fourier_terms_use_each_row_time(self):
        times = torch.tensor([[0.0], [0.5]])
        params = torch.tensor([[[2.0, 3.0]], [[2.0, 3.0]]])

        values = get_pointwise_func_result(times, params, [0, 0, 0, 1, 0, 0])

        torch.testing.assert_close(values[:, 0], torch.tensor([3.0, 2.0]), atol=1e-6, rtol=1e-6)

    def test_quaternion_spline_uses_each_row_time(self):
        angles = torch.tensor([0.0, math.pi / 2.0, math.pi])
        controls = torch.stack(
            [torch.cos(angles / 2.0), torch.zeros(3), torch.zeros(3), torch.sin(angles / 2.0)],
            dim=1,
        )
        params = (controls - torch.tensor([1.0, 0.0, 0.0, 0.0]))
        params = params.T.unsqueeze(0).repeat(3, 1, 1)
        times = torch.tensor([[0.25], [0.50], [1.00]])

        values = get_pointwise_func_result(times, params, [0, 0, 0, 0, 3, 1])

        expected_angles = torch.tensor([math.pi / 4.0, math.pi / 2.0, math.pi])
        expected = torch.stack(
            [
                torch.cos(expected_angles / 2.0),
                torch.zeros(3),
                torch.zeros(3),
                torch.sin(expected_angles / 2.0),
            ],
            dim=1,
        )
        alignment = torch.abs(torch.sum(values * expected, dim=-1))
        torch.testing.assert_close(alignment, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_relative_rotation_is_identity_at_anchor(self):
        torch.manual_seed(7)
        params = torch.randn(4, 4, 5) * 0.1
        times = torch.tensor([[0.0], [0.2], [0.7], [1.0]])
        anchor = get_pointwise_func_result(times, params, [0, 0, 0, 0, 5, 2])

        relative = quaternion_multiply(anchor, quaternion_conjugate(anchor))

        expected = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(relative)
        torch.testing.assert_close(relative, expected, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "baseline evaluator requires CUDA")
    def test_matches_baseline_scalar_evaluator(self):
        torch.manual_seed(11)
        times = torch.tensor([[0.0], [0.2], [0.7], [1.0]], device="cuda")

        translation_args = [5, 2, 2, 2, 0, 0]
        translation_params = torch.randn(4, 3, 11, device="cuda")
        pointwise_translation = get_pointwise_func_result(
            times, translation_params, translation_args
        )
        scalar_translation = torch.stack(
            [
                get_func_result(
                    times[index].item(),
                    translation_params[index:index + 1],
                    translation_args,
                )[0]
                for index in range(len(times))
            ]
        )
        torch.testing.assert_close(pointwise_translation, scalar_translation)

        rotation_args = [0, 0, 0, 0, 5, 2]
        rotation_params = torch.randn(4, 4, 5, device="cuda") * 0.1
        pointwise_rotation = get_pointwise_func_result(times, rotation_params, rotation_args)
        scalar_rotation = torch.stack(
            [
                get_func_result(
                    times[index].item(),
                    rotation_params[index:index + 1],
                    rotation_args,
                )[0]
                for index in range(len(times))
            ]
        )
        alignment = torch.abs(torch.sum(pointwise_rotation * scalar_rotation, dim=-1))
        torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
