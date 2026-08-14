import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.trust4d.cache_grounding_dino import sha256


class CacheGroundingDinoTest(unittest.TestCase):
    def test_sha256_matches_content(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.safetensors"
            path.write_bytes(b"fixed grounding model")
            self.assertEqual(sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
