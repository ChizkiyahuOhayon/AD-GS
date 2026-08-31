import copy
import tempfile
import unittest

import numpy as np
import torch


class GaugeFixLifecycleTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires the AD-GS CUDA environment")
    def test_chart_and_actor_set_survive_full_checkpoint(self):
        from arguments.waymo import order_args
        from models.road_chart import BicubicRoadChart
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import BasicPointCloud

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            GaussianModel(
                sh_degree=0,
                order_args=copy.deepcopy(order_args),
                oracle_contact=True,
                gauge_fix=True,
            )

        point_cloud = BasicPointCloud(
            points=np.array(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 0.5], [3.0, 2.0, 0.5]],
                dtype=np.float32,
            ),
            colors=np.full((3, 3), 0.5, dtype=np.float32),
            normals=np.zeros((3, 3), dtype=np.float32),
            time=np.array([[-1.0], [0.0], [0.0]], dtype=np.float32),
            obj_id=np.array([[0.0], [7.0], [7.0]], dtype=np.float32),
        )
        model = GaussianModel(
            sh_degree=0,
            order_args=copy.deepcopy(order_args),
            gauge_fix=True,
        )
        model.create_from_pcd(
            point_cloud,
            scene_extent=10.0,
            cameras_extent=10.0,
            frame_gap=0.1,
            default_order_downsample_ratio=3.0,
        )
        model.road_chart = BicubicRoadChart(
            torch.full((5, 6), 0.25, device="cuda"),
            torch.tensor([-2.0, -2.0], device="cuda"),
            2.0,
        ).cuda()
        model.road_chart.control_heights.requires_grad_(False)
        model.gauge_actor_ids = torch.tensor([7], device="cuda")
        model.gauge_fix_metadata = {"actor_count": 1}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = directory + "/point_cloud.ply"
            model.save_ply(checkpoint_path)
            restored = GaussianModel(
                sh_degree=0,
                order_args=copy.deepcopy(order_args),
                gauge_fix=True,
            )
            restored.load_ply(checkpoint_path)

        self.assertTrue(
            torch.equal(
                restored.road_chart.control_heights,
                model.road_chart.control_heights,
            )
        )
        self.assertTrue(torch.equal(restored.gauge_actor_ids, model.gauge_actor_ids))
        self.assertEqual(restored.gauge_fix_metadata, model.gauge_fix_metadata)
        self.assertFalse(restored.road_chart.control_heights.requires_grad)


if __name__ == "__main__":
    unittest.main()
