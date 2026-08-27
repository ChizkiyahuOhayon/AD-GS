import torch


def gradient_persistence_weight(gradient_sum, gradient_sq_sum, count, gamma):
    """Return mean-one weights that favor consistent multi-view gradients."""
    if gamma <= 0:
        return torch.ones_like(gradient_sum)

    valid = count > 1
    if not torch.any(valid):
        return torch.ones_like(gradient_sum)

    safe_count = count.clamp_min(1)
    gradient_mean = gradient_sum / safe_count
    gradient_sq_mean = gradient_sq_sum / safe_count
    persistence = gradient_mean.square() / gradient_sq_mean.clamp_min(1e-12)
    weights = persistence.clamp(0, 1).pow(gamma)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return weights / weights[valid].mean().clamp_min(1e-12)
