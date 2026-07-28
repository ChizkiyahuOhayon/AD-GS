import copy
import tempfile
import unittest
from argparse import ArgumentParser

import numpy as np
import torch


class ActorIdLifecycleTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires the AD-GS CUDA environment")
    def test_actor_ids_survive_initialization_densification_pruning_and_checkpoint(self):
        from arguments import OptimizationParams
        from arguments.waymo import order_args
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import BasicPointCloud

        point_cloud = BasicPointCloud(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [1.2, 0.0, 0.2], [2.0, 0.0, 0.3]],
                dtype=np.float32,
            ),
            colors=np.full((4, 3), 0.5, dtype=np.float32),
            normals=np.zeros((4, 3), dtype=np.float32),
            time=np.array([[-1.0], [0.0], [0.0], [0.0]], dtype=np.float32),
            obj_id=np.array([[0.0], [11.0], [11.0], [22.0]], dtype=np.float32),
        )
        model = GaussianModel(sh_degree=0, order_args=copy.deepcopy(order_args))
        model.create_from_pcd(
            point_cloud,
            scene_extent=10.0,
            cameras_extent=10.0,
            frame_gap=0.1,
            default_order_downsample_ratio=3.0,
        )
        self.assertEqual(model.get_obj_actor_id.tolist(), [11, 11, 22])

        model._scene_scaling.data.fill_(-5.0)
        model._obj_scaling.data.fill_(-5.0)
        parser = ArgumentParser()
        optimization = OptimizationParams(parser, config={}).extract(parser.parse_args([]))
        optimization.lambda_reg = 0.0
        optimization.lambda_sigma = 0.0
        model.training_setup(optimization)

        model.densify_and_clone(
            torch.zeros(model.get_scene_pts_num, dtype=torch.bool, device='cuda'),
            torch.tensor([True, False, False], dtype=torch.bool, device='cuda'),
        )
        self.assertEqual(model.get_obj_actor_id.tolist(), [11, 11, 22, 11])

        model.prune_points(
            torch.zeros(model.get_scene_pts_num, dtype=torch.bool, device='cuda'),
            torch.tensor([False, True, False, False], dtype=torch.bool, device='cuda'),
        )
        self.assertEqual(model.get_obj_actor_id.tolist(), [11, 22, 11])

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = directory + "/point_cloud.ply"
            model.save_ply(checkpoint_path)
            restored = GaussianModel(sh_degree=0, order_args=copy.deepcopy(order_args))
            restored.load_ply(checkpoint_path)
            self.assertEqual(restored.get_obj_actor_id.tolist(), [11, 22, 11])


if __name__ == "__main__":
    unittest.main()
