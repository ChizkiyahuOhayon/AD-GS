import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.trust4d.cache_adgs_evaluator import sha256, validate_digest


class CacheAdgsEvaluatorTest(unittest.TestCase):
    def test_sha256_matches_content(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "weight.pth"
            path.write_bytes(b"fixed evaluator weight")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(sha256(path), expected)
            self.assertEqual(validate_digest(path, expected), expected)

    def test_checksum_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "weight.pth"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_digest(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
