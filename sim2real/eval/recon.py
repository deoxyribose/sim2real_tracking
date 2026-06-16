"""Reconstruction quality metrics: PSNR and SSIM."""

from __future__ import annotations

import jax.numpy as jnp


def psnr(pred, target, data_range: float = 1.0) -> float:
    mse = jnp.mean((pred - target) ** 2)
    return float(20.0 * jnp.log10(data_range) - 10.0 * jnp.log10(mse + 1e-12))


def ssim_simple(pred, target, *, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> float:
    """Single-window SSIM. Pred and target must have shape (..., H, W, C).

    Uses no Gaussian window — simple mean-and-variance based. Good enough for a coarse signal.
    """
    mu_x = jnp.mean(pred)
    mu_y = jnp.mean(target)
    var_x = jnp.var(pred)
    var_y = jnp.var(target)
    cov = jnp.mean((pred - mu_x) * (target - mu_y))
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / (den + 1e-12))
