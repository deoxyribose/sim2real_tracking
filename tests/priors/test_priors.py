"""Tests for prior log_prob / KL semantics."""

import jax.numpy as jnp

from sim2real.priors import where as where_mod
from sim2real.priors import pres as pres_mod
from sim2real.priors import what as what_mod
from sim2real.priors import style as style_mod


def test_where_log_prob_zero_delta_is_finite():
    cfg = where_mod.WherePriorConfig()
    zw = jnp.zeros((3, 4, 3))
    alive = jnp.ones((3, 4))
    lp = where_mod.log_prob(zw, alive, cfg)
    assert jnp.isfinite(lp)


def test_where_log_prob_increases_with_smaller_delta():
    cfg = where_mod.WherePriorConfig()
    alive = jnp.ones((2, 4))
    zw_small = jnp.zeros((2, 4, 3))
    zw_big = zw_small.at[1].set(jnp.ones((4, 3)))
    lp_small = where_mod.log_prob(zw_small, alive, cfg)
    lp_big = where_mod.log_prob(zw_big, alive, cfg)
    assert float(lp_small) > float(lp_big)


def test_pres_birth_zero_when_discovery_zero():
    cfg = pres_mod.PresPriorConfig(p_init=0.1, p_keep=0.9, p_birth_max=0.5)
    # All slots dead at t=0, all dead at t=1, discovery=0 → birth prob = 0 → log_prob is high.
    zp = jnp.zeros((2, 3))
    disc = jnp.zeros((2, 3))
    lp = pres_mod.log_prob(zp, disc, cfg)
    # Compare with discovery=1.0 → birth=0.5, log(1-0.5) per slot.
    disc_full = jnp.ones((2, 3))
    lp_full = pres_mod.log_prob(zp, disc_full, cfg)
    assert float(lp) > float(lp_full)


def test_what_temporal_kl_is_zero_when_posterior_matches_prior_exactly():
    cfg = what_mod.WhatPriorConfig(sigma_init=1.0, sigma_step=0.5, alpha=1.0)
    T, N, Zw = 3, 2, 8
    # Set posterior to match the prior:
    # at t=0: N(0, 1²) → mu=0, logvar=0.
    # at t>=1: N(alpha · mu_{t-1}, σ_step²) → with mu_q = 0 throughout, prev_mu_q = 0, matches.
    mu_q = jnp.zeros((T, N, Zw))
    logvar_q = jnp.concatenate(
        [jnp.zeros((1, N, Zw)), jnp.broadcast_to(jnp.log(0.25), (T - 1, N, Zw))], axis=0
    )
    prev_mu_q = jnp.zeros_like(mu_q)
    alive = jnp.ones((T, N))
    kl = what_mod.kl_temporal(mu_q, logvar_q, prev_mu_q, alive, cfg)
    assert abs(float(kl)) < 1e-4


def test_what_cross_kl_finite():
    cfg = what_mod.WhatPriorConfig()
    mu_q = jnp.zeros((3, 4, 8))
    logvar_q = jnp.zeros((3, 4, 8))
    alive = jnp.ones((3, 4))
    kl = what_mod.kl_cross_object(mu_q, logvar_q, alive, cfg)
    assert jnp.isfinite(kl)


def test_style_kl_zero_for_standard_normal_posterior():
    mu = jnp.zeros(8)
    logvar = jnp.zeros(8)
    assert abs(float(style_mod.kl(mu, logvar))) < 1e-5


if __name__ == "__main__":
    import inspect, sys
    failed = 0
    for name, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}")
            failed += 1
    sys.exit(failed)
