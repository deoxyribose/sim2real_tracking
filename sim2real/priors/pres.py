"""z_pres prior: 2-state Markov chain with discovery-conditional birth.

p(pres_t=1 | pres_{t-1}=1) = p_keep
p(pres_t=1 | pres_{t-1}=0) = p_birth_max × discovery_score    (discovery_score ∈ [0,1])
p(pres_t=0 | pres_{t-1}=1) = 1 - p_keep
p(pres_t=0 | pres_{t-1}=0) = 1 - p_birth_max × discovery_score

At t=0, prior is Bernoulli(p_init).

The discovery score gates birth — if the model proposes a slot in an already-explained image
region, birth is suppressed. The score is the residual-mask integral at the slot's z_where
location and is computed in the model (model.aux).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

Array = jnp.ndarray


@dataclass(frozen=True)
class PresPriorConfig:
    p_init: float = 0.1
    p_keep: float = 0.95
    p_birth_max: float = 0.05


def _bernoulli_log_prob(p, x, eps=1e-6):
    p = jnp.clip(p, eps, 1.0 - eps)
    return x * jnp.log(p) + (1.0 - x) * jnp.log(1.0 - p)


def log_prob(
    z_pres: Array,
    discovery_score: Array | None,
    cfg: PresPriorConfig,
) -> Array:
    """
    Args:
      z_pres: (T, N) in {0,1} (soft values in [0,1] from Gumbel-sigmoid are accepted too).
      discovery_score: (T, N) in [0,1] — if None, uses 1.0 (always allow birth).

    Returns: scalar sum over (T, N).
    """
    T = z_pres.shape[0]
    if discovery_score is None:
        discovery_score = jnp.ones_like(z_pres)

    # t=0: Bernoulli(p_init)
    lp0 = jnp.sum(_bernoulli_log_prob(jnp.full_like(z_pres[0], cfg.p_init), z_pres[0]))

    # t>=1
    z_t = z_pres[1:]                                                                 # (T-1, N)
    z_tm1 = z_pres[:-1]
    disc_t = discovery_score[1:]
    p_birth = cfg.p_birth_max * disc_t
    p_t = z_tm1 * cfg.p_keep + (1.0 - z_tm1) * p_birth
    lp_t = jnp.sum(_bernoulli_log_prob(p_t, z_t))
    return lp0 + lp_t


def kl(z_pres: Array, z_pres_logit: Array, discovery_score: Array | None,
       cfg: PresPriorConfig) -> Array:
    """KL between a Bernoulli posterior (logit `z_pres_logit`) and the Markov prior.

    Treats the posterior as Bernoulli(σ(logit)); the KL term decomposes per step:
      KL(Bern(q_t) || Bern(p_t))   where p_t = p_keep · z_{t-1} + (1-z_{t-1}) · p_birth · disc.
    We use the *expected* z_{t-1} from the posterior (passed in `z_pres`) — same level of
    approximation as cellulose.
    """
    import jax
    q = jax.nn.sigmoid(z_pres_logit)
    if discovery_score is None:
        discovery_score = jnp.ones_like(q)
    # t=0 prior
    p0 = jnp.full_like(q[0], cfg.p_init)
    # t>=1 prior
    p_birth = cfg.p_birth_max * discovery_score[1:]
    p_rest = z_pres[:-1] * cfg.p_keep + (1.0 - z_pres[:-1]) * p_birth
    p = jnp.concatenate([p0[None], p_rest], axis=0)                                  # (T, N)
    eps = 1e-6
    p = jnp.clip(p, eps, 1.0 - eps)
    q = jnp.clip(q, eps, 1.0 - eps)
    return jnp.sum(q * (jnp.log(q) - jnp.log(p)) + (1.0 - q) * (jnp.log(1.0 - q) - jnp.log(1.0 - p)))
