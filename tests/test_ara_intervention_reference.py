import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "research/ara/trust4d_teacher_reliability/src/execution/intervention_reference.py"
)
SPEC = importlib.util.spec_from_file_location("intervention_reference", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _extrinsics(rotations, centers):
    translations = -torch.einsum("nij,nj->ni", rotations, centers)
    return torch.cat((rotations, translations.unsqueeze(-1)), dim=-1)


def test_camera_guided_similarity_recovers_exact_transform():
    dtype = torch.float64
    angle = torch.tensor(0.37, dtype=dtype)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    scale = torch.tensor(2.4, dtype=dtype)
    translation = torch.tensor([1.2, -0.7, 3.1], dtype=dtype)
    target_rotations = torch.eye(3, dtype=dtype).repeat(4, 1, 1)
    target_centers = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, -0.1, 0.3], [3.0, 0.4, -0.2]],
        dtype=dtype,
    )
    predicted_rotations = target_rotations @ rotation
    predicted_centers = (target_centers - translation) @ rotation / scale

    fit = MODULE.fit_camera_guided_similarity(
        _extrinsics(predicted_rotations, predicted_centers),
        _extrinsics(target_rotations, target_centers),
    )

    assert torch.allclose(fit.rotation, rotation, atol=1e-10)
    assert torch.allclose(fit.scale, scale, atol=1e-10)
    assert torch.allclose(fit.translation, translation, atol=1e-10)
    assert fit.rotation_residual_deg < 1e-5
    assert fit.center_rmse < 1e-10


def test_intervention_statistics_recovers_error_direction():
    displacements = torch.tensor(
        [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    stats = MODULE.intervention_statistics(displacements, torch.tensor([3.0, 0.0, 0.0]))

    assert torch.allclose(stats.disagreement, torch.tensor(2.0))
    assert torch.allclose(stats.directional_capture, torch.tensor(1.0))
    assert torch.allclose(stats.principal_direction.abs(), torch.tensor([1.0, 0.0, 0.0]))


def test_similarity_rejects_degenerate_centers():
    rotations = torch.eye(3).repeat(2, 1, 1)
    centers = torch.zeros(2, 3)
    extrinsics = _extrinsics(rotations, centers)

    try:
        MODULE.fit_camera_guided_similarity(extrinsics, extrinsics)
    except ValueError as error:
        assert "constrain scale" in str(error)
    else:
        raise AssertionError("degenerate centers must be rejected")
