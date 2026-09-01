"""Training schedule for gradually introducing GF-DGS contact."""


def linear_contact_strength(
    iteration,
    total_iterations,
    warmup_fraction=0.0,
    ramp_fraction=0.0,
):
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if (
        warmup_fraction < 0.0
        or ramp_fraction < 0.0
        or warmup_fraction + ramp_fraction > 1.0
    ):
        raise ValueError("contact schedule fractions must be non-negative and sum to <= 1")

    progress = iteration / total_iterations
    if progress <= warmup_fraction:
        return 0.0
    if ramp_fraction == 0.0:
        return 1.0
    return min((progress - warmup_fraction) / ramp_fraction, 1.0)
