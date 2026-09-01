import tempfile
import unittest
from pathlib import Path

from scripts.waymo.inventory_preprocessed import (
    CRITICAL_FILES,
    MODALITIES,
    SCENES,
    inventory_waymo,
)


def create_dataset(root):
    for scene in SCENES:
        scene_root = root / scene
        for modality in MODALITIES:
            path = scene_root / modality / "frame.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes("{}:{}".format(scene, modality).encode("utf-8"))
        for name in CRITICAL_FILES:
            (scene_root / name).write_bytes("{}:{}".format(scene, name).encode("utf-8"))


class WaymoInventoryTest(unittest.TestCase):
    def test_complete_datasets_produce_identical_receipts(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            create_dataset(first_root)
            create_dataset(second_root)

            first_receipt = inventory_waymo(first_root)
            second_receipt = inventory_waymo(second_root)

        self.assertTrue(first_receipt["passed"])
        self.assertEqual(first_receipt, second_receipt)
        scene = first_receipt["scenes"]["scene090"]
        self.assertEqual(scene["modalities"]["image"]["files"], 1)
        self.assertEqual(len(scene["critical_files"]["cameras.npz"]["sha256"]), 64)

    def test_missing_critical_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_dataset(root)
            (root / "scene090" / "cameras.npz").unlink()
            receipt = inventory_waymo(root)

        self.assertFalse(receipt["passed"])
        self.assertIn(
            "scene090/cameras.npz is missing or empty",
            receipt["scenes"]["scene090"]["errors"],
        )

    def test_empty_modality_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_dataset(root)
            (root / "scene026" / "flow" / "frame.bin").unlink()
            receipt = inventory_waymo(root)

        self.assertFalse(receipt["passed"])
        self.assertIn(
            "scene026/flow contains no files",
            receipt["scenes"]["scene026"]["errors"],
        )


if __name__ == "__main__":
    unittest.main()
