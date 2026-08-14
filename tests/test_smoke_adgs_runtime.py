import unittest

from scripts.trust4d.smoke_adgs_runtime import EXPECTED_VERSIONS, check_version_contract


class SmokeAdgsRuntimeTest(unittest.TestCase):
    def test_exact_versions_pass(self):
        check_version_contract(EXPECTED_VERSIONS)

    def test_cuda_suffix_is_allowed(self):
        actual = dict(EXPECTED_VERSIONS, torch="1.13.1+cu117")
        check_version_contract(actual)

    def test_version_mismatch_fails(self):
        actual = dict(EXPECTED_VERSIONS, pytorch3d="0.7.9")
        with self.assertRaisesRegex(ValueError, "pytorch3d"):
            check_version_contract(actual)


if __name__ == "__main__":
    unittest.main()
