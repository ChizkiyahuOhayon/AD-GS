import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "analysis" / "manifests" / "waymo_adgs_8.json"
BASELINE_PROTOCOL = (
    ROOT / "analysis" / "manifests" / "adgs_waymo_baseline_protocol.json"
)
WAYMO_SCRIPTS = ROOT / "scripts" / "waymo"
DOWNLOAD_SCRIPT = WAYMO_SCRIPTS / "download_manifest.py"
DOWNLOAD_SPEC = importlib.util.spec_from_file_location("download_manifest", DOWNLOAD_SCRIPT)
DOWNLOAD_MODULE = importlib.util.module_from_spec(DOWNLOAD_SPEC)
DOWNLOAD_SPEC.loader.exec_module(DOWNLOAD_MODULE)
VERIFY_SCRIPT = WAYMO_SCRIPTS / "verify_baseline.py"
VERIFY_SPEC = importlib.util.spec_from_file_location("verify_baseline", VERIFY_SCRIPT)
VERIFY_MODULE = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(VERIFY_MODULE)


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
        expected = [
            {
                key: sequence[key]
                for key in ("scene", "first_frame", "last_frame", "filename")
            }
            for sequence in self.sequences
        ]
        self.assertEqual(commands, expected)

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

    def test_downloader_builds_exact_v141_object_list(self):
        objects = DOWNLOAD_MODULE.load_objects(MANIFEST)
        self.assertEqual([item["scene"] for item in objects], self.expected_scenes)
        self.assertEqual(
            [item["filename"] for item in objects],
            [sequence["filename"] for sequence in self.sequences],
        )
        self.assertTrue(
            all(
                item["uri"].startswith(
                    DOWNLOAD_MODULE.EXPECTED_PREFIX + "/validation/segment-"
                )
                for item in objects
            )
        )
        self.assertEqual(
            [item["download_name"] for item in objects],
            [Path(sequence["gcs_object"]).name for sequence in self.sequences],
        )

    def test_baseline_protocol_locks_official_eight_scene_run(self):
        protocol = json.loads(BASELINE_PROTOCOL.read_text())
        self.assertEqual(protocol["scenes"], self.expected_scenes)
        self.assertEqual(
            protocol["upstream_commit"],
            "9a208512e49c8ddbaa20387921d9648adcd21cb4",
        )
        self.assertEqual(
            protocol["upstream_repository"],
            "https://github.com/JiaweiXu8/AD-GS.git",
        )
        self.assertEqual(protocol["iterations"], 60_000)
        self.assertEqual(protocol["seed"], 0)
        self.assertTrue(protocol["hard_gate"]["all_scenes_required"])
        self.assertFalse(
            protocol["hard_gate"]["method_integration_allowed_before_pass"]
        )

    def test_baseline_auditor_checks_exact_split_and_artifacts(self):
        protocol = json.loads(BASELINE_PROTOCOL.read_text())
        scene = protocol["scenes"][0]
        iteration = protocol["iterations"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source" / scene
            output = root / "output" / scene
            logs = root / "logs"
            source.mkdir(parents=True)
            logs.mkdir()
            is_val = np.asarray([False, False, False, False, True, False])
            np.savez(source / "cameras.npz", is_val_list=is_val)
            for relative in ("cfg_args", "cameras.json", "input.ply"):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("iterations={}\n".format(iteration))
            checkpoint = output / "point_cloud" / "iteration_{}".format(iteration)
            checkpoint.mkdir(parents=True)
            (checkpoint / "point_cloud.ply").write_text("ply")
            (checkpoint / "env.pth").write_text("weights")
            for split, count in (("test", 1), ("train", 5)):
                for folder in ("renders", "gt"):
                    path = output / split / "ours_{}".format(iteration) / folder
                    path.mkdir(parents=True)
                    for index in range(count):
                        (path / "{:05d}.png".format(index)).write_text("png")
            metrics = {
                "ours_{}".format(iteration): {
                    name: 1.0 for name in protocol["hard_gate"]["required_metrics"]
                }
            }
            (output / "results.json").write_text(json.dumps(metrics))
            (output / "results-train.json").write_text(json.dumps(metrics))
            (logs / "{}.train.log".format(scene)).write_text("Training complete.\n")
            (logs / "{}.render.log".format(scene)).write_text("LPIPS(VGG)\n")
            (logs / "{}.train.exitcode".format(scene)).write_text("0\n")
            (logs / "{}.render.exitcode".format(scene)).write_text("0\n")

            audit = VERIFY_MODULE.audit_scene(
                protocol, scene, root / "source", root / "output", logs
            )
            self.assertTrue(audit["passed"], audit["errors"])
            (checkpoint / "env.pth").unlink()
            audit = VERIFY_MODULE.audit_scene(
                protocol, scene, root / "source", root / "output", logs
            )
            self.assertFalse(audit["passed"])


if __name__ == "__main__":
    unittest.main()
