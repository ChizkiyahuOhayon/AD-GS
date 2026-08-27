import unittest

import torch

from utils.densification_utils import gradient_persistence_weight


class GradientPersistenceWeightTest(unittest.TestCase):
    def test_prefers_consistent_evidence_over_equal_mean_spike(self):
        gradient_sum = torch.tensor([[4.0], [4.0]])
        gradient_sq_sum = torch.tensor([[8.0], [16.0]])
        count = torch.tensor([[2.0], [2.0]])

        weights = gradient_persistence_weight(
            gradient_sum, gradient_sq_sum, count, gamma=1.0
        )

        self.assertGreater(weights[0].item(), weights[1].item())
        torch.testing.assert_close(weights.mean(), torch.tensor(1.0))

    def test_requires_repeated_observations_when_available(self):
        gradient_sum = torch.tensor([[4.0], [100.0]])
        gradient_sq_sum = torch.tensor([[8.0], [10_000.0]])
        count = torch.tensor([[2.0], [1.0]])

        weights = gradient_persistence_weight(
            gradient_sum, gradient_sq_sum, count, gamma=1.0
        )

        torch.testing.assert_close(weights, torch.tensor([[1.0], [0.0]]))

    def test_falls_back_to_baseline_without_repeated_observations(self):
        weights = gradient_persistence_weight(
            torch.tensor([[2.0], [3.0]]),
            torch.tensor([[4.0], [9.0]]),
            torch.ones(2, 1),
            gamma=1.0,
        )

        torch.testing.assert_close(weights, torch.ones(2, 1))

    def test_zero_gamma_is_exact_baseline_weight(self):
        weights = gradient_persistence_weight(
            torch.tensor([[2.0], [3.0]]),
            torch.tensor([[4.0], [9.0]]),
            torch.tensor([[2.0], [3.0]]),
            gamma=0.0,
        )

        torch.testing.assert_close(weights, torch.ones(2, 1))


if __name__ == "__main__":
    unittest.main()
