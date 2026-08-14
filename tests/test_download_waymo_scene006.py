import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "trust4d" / "download_waymo_scene006.sh"
EXPECTED_URI = (
    "gs://waymo_open_dataset_v_1_4_1/individual_files/validation/"
    "segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord"
)
EXPECTED_FILENAME = (
    "individual_files_validation_segment-10448102132863604198_472_000_492_000_"
    "with_camera_labels.tfrecord"
)


class DownloadWaymoScene006Test(unittest.TestCase):
    def test_remote_object_and_local_filename_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_bin = root / "bin"
            destination = root / "raw"
            log = root / "gcloud.log"
            fake_bin.mkdir()
            fake_gcloud = fake_bin / "gcloud"
            fake_gcloud.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$TEST_GCLOUD_LOG\"\n"
                "if [[ \"$1 $2\" == 'auth print-access-token' ]]; then\n"
                "  printf 'test-token\\n'\n"
                "elif [[ \"$1 $2 $3\" == 'storage objects describe' ]]; then\n"
                "  if [[ \"$5\" == '--format=value(size)' ]]; then\n"
                "    printf '3\\n'\n"
                "  else\n"
                "    printf '{\"size\": \"3\"}\\n'\n"
                "  fi\n"
                "elif [[ \"$1 $2\" == 'storage cp' ]]; then\n"
                "  printf 'abc' > \"$4\"\n"
                "else\n"
                "  exit 64\n"
                "fi\n"
            )
            fake_gcloud.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["TEST_GCLOUD_LOG"] = str(log)

            subprocess.run(
                ["bash", str(SCRIPT), str(destination)],
                cwd=REPOSITORY,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

            calls = log.read_text()
            self.assertIn(EXPECTED_URI, calls)
            self.assertNotIn(
                "/individual_files/individual_files_validation_segment-", calls
            )
            self.assertEqual((destination / EXPECTED_FILENAME).read_bytes(), b"abc")


if __name__ == "__main__":
    unittest.main()
