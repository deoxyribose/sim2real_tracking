"""Optimizer schedules + KL ramp."""

from __future__ import annotations

import optax


def adamw_cosine(lr_peak: float, total_steps: int, warmup_steps: int = 500, wd: float = 1e-4):
    """Cosine LR schedule with warmup. Returns (optimizer, schedule_fn).

    Note: optax's `warmup_cosine_decay_schedule` interprets `decay_steps` as the TOTAL number
    of steps over the full run (warmup + cosine decay), not the post-warmup duration.
    """
    warmup_steps = max(min(warmup_steps, max(total_steps - 1, 1)), 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=lr_peak * 0.05,
    )
    return optax.adamw(schedule, weight_decay=wd), schedule


def kl_ramp(step: int, ramp_steps: int = 2000, max_beta: float = 1.0) -> float:
    """Linear ramp from 0 → max_beta over `ramp_steps`."""
    if ramp_steps <= 0:
        return max_beta
    return float(min(step / ramp_steps, 1.0)) * max_beta
