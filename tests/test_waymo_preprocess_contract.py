import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "analysis" / "manifests" / "waymo_adgs_8.json"
WAYMO_SCRIPTS = ROOT / "scripts" / "waymo"


class WaymoPreprocessContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sequences = json.loads(MANIFEST.read_text())["sequences"]
        cls.expected_scenes = [sequence["scene"] for sequence in cls.sequences]

    def test_tfrecord_commands_match_manifest(self):
        command_pattern = re.compile(
            r"python scripts/waymo/waymo\.py \$1/(?P<filename>\S+) "
            r"\./data/waymo/(?P<scene>scene\d+) "
            r"--first_frame (?P<first_frame>\d+) "
            r"--last_frame (?P<last_frame>\d+) --use_color"
        )
        commands = [
            {
                "scene": match.group("scene"),
                "first_frame": int(match.group("first_frame")),
                "last_frame": int(match.group("last_frame")),
                "filename": match.group("filename"),
            }
            for match in command_pattern.finditer(
                (WAYMO_SCRIPTS / "prepare-waymo.sh").read_text()
            )
        ]
        self.assertEqual(commands, self.sequences)

    def test_all_batch_scripts_cover_exact_scene_set(self):
        for script_name in (
            "prepare-flow.sh",
            "prepare-colmap.sh",
            "segment-pcd.sh",
            "run-waymo.sh",
        ):
            text = (WAYMO_SCRIPTS / script_name).read_text()
            actual_scenes = list(dict.fromkeys(re.findall(r"scene\d+", text)))
            self.assertEqual(actual_scenes, self.expected_scenes, script_name)

    def test_flow_commands_require_pinned_local_cotracker(self):
        text = (WAYMO_SCRIPTS / "prepare-flow.sh").read_text()
        flow_commands = [
            line for line in text.splitlines() if line.startswith("python scripts/flow.py")
        ]
        self.assertEqual(len(flow_commands), len(self.expected_scenes))
        for command in flow_commands:
            self.assertTrue(command.endswith('--cotracker_repo "$1"'), command)


if __name__ == "__main__":
    unittest.main()
