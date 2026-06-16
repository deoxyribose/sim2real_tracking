"""Prior interface shared by all latents.

Every prior provides:
  - log_prob(latent_t, latent_tm1=None, conditioning=None) → scalar (sum over slots/dims).
  - kl(q_mu, q_logvar, ...) → scalar, when applicable.
  - sample(key, prev=None, conditioning=None) → latent_t — used by ancestral sampling tests.

All priors operate on a single video (no batch dim); the caller vmaps over the batch.
"""

from __future__ import annotations

import jax.numpy as jnp


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """KL(N(mu_q, σ_q) || N(mu_p, σ_p)). Both args broadcast."""
    var_q = jnp.exp(logvar_q)
    var_p = jnp.exp(logvar_p)
    return 0.5 * jnp.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0,
        axis=-1,
    )
