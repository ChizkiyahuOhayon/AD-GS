import unittest

import numpy as np

from scripts.trust4d.smoke_waymo_prior_runtime import (
    EXPECTED,
    check_versions,
    tensor_record,
)


class SmokeWaymoPriorRuntimeTest(unittest.TestCase):
    def test_exact_versions_pass(self):
        for stage, versions in EXPECTED.items():
            check_versions(stage, versions)

    def test_cuda_suffix_passes(self):
        actual = dict(EXPECTED["dpt"], torch="2.4.1+cu118")
        check_versions("dpt", actual)

    def test_wrong_version_fails(self):
        actual = dict(EXPECTED["flow"], timm="1.0.8")
        with self.assertRaisesRegex(ValueError, "timm"):
            check_versions("flow", actual)

    def test_tensor_record_rejects_nonfinite(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable locally")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            tensor_record(torch.tensor([np.nan]))


if __name__ == "__main__":
    unittest.main()
