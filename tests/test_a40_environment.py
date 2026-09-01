import unittest

from scripts.server.verify_a40_environment import EXPECTED, validate_runtime


class A40EnvironmentTest(unittest.TestCase):
    def test_locked_runtime_is_accepted(self):
        record = dict(EXPECTED)
        record.update(gpu_name="NVIDIA A40", compute_capability=[8, 6])
        self.assertEqual(validate_runtime(record), [])

    def test_mismatches_are_reported_together(self):
        record = dict(EXPECTED)
        record.update(
            torch="2.4.1",
            torch_cuda="12.1",
            gpu_name="NVIDIA GeForce RTX 4090",
            compute_capability=[8, 9],
        )
        errors = validate_runtime(record)
        self.assertEqual(len(errors), 4)
        self.assertTrue(any("torch is 2.4.1" in error for error in errors))
        self.assertTrue(any("expected NVIDIA A40" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
