import torch


def corrective_pair_samples(
    scales, rotations, exp_avg, exp_avg_sq, fallback_samples, adam_eps
):
    """Place one split child at its parent and one along Adam's descent direction."""
    eps = torch.finfo(scales.dtype).eps
    direction = -exp_avg / (
        torch.sqrt(torch.clamp_min(exp_avg_sq, 0.0)) + adam_eps
    )
    local_direction = torch.bmm(
        rotations.transpose(1, 2), direction.unsqueeze(-1)
    ).squeeze(-1)
    mahalanobis_norm = torch.linalg.vector_norm(
        local_direction / scales, dim=-1, keepdim=True
    )
    corrective = local_direction / mahalanobis_norm.clamp_min(eps)

    valid = (
        torch.isfinite(corrective).all(dim=-1, keepdim=True)
        & torch.isfinite(mahalanobis_norm)
        & (mahalanobis_norm > eps)
        & torch.isfinite(scales).all(dim=-1, keepdim=True)
        & (scales > eps).all(dim=-1, keepdim=True)
    )
    samples = torch.cat([torch.zeros_like(corrective), corrective], dim=0)
    return torch.where(valid.repeat(2, 1), samples, fallback_samples)
