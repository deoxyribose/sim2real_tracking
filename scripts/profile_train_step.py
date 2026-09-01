"""Profile each part of the training loop separately: sim only,
forward only, forward+backward, full train_step. All timings after
JIT warmup."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sim2real.model.energy_loss import EnergyLossConfig, compute_energy_loss
from sim2real.model.unet_energy import (
    UNetConfig, UNetEnergy, decode_curves, sample_batched_noise, unpack_pred,
)
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip
from sim2real.scripts.train_energy import load_pca

BATCH = 24
T = 16
H = 256

print(f"device: {jax.devices()}   BATCH={BATCH}   H={H}   T={T}", flush=True)

cfg_u = UNetConfig(T=T, H=H, W=H)
cfg_e = EnergyLossConfig(beta=0.05)
sim_cfg = DiverseSimConfig(T=T, H=H, W=H)

# ---- sim only ----
sample_sim_batch = jax.jit(jax.vmap(lambda k: sample_clip(k, sim_cfg)))

def sim_only(n_batches=10):
    keys = jax.random.split(jax.random.key(0), BATCH)
    _ = sample_sim_batch(keys); _["clip_raw"].block_until_ready()
    t0 = time.time()
    for i in range(n_batches):
        out = sample_sim_batch(jax.random.split(jax.random.key(i), BATCH))
        out["clip_raw"].block_until_ready()
    return (time.time() - t0) / n_batches

# ---- model forward only ----
pca_mean, pca_basis = load_pca("data_cache/flagella_pca.npz")
model = UNetEnergy(cfg=cfg_u)
key = jax.random.key(0)
video = jnp.zeros((BATCH, T, H, H))
noise = jnp.zeros((BATCH, H, H, 1))
params = model.init(key, video, noise, train=True)

@jax.jit
def forward_fn(params, video, noise):
    return model.apply(params, video, noise, train=False)

def fwd_only(n=10):
    _ = forward_fn(params, video, noise); _.block_until_ready()
    t0 = time.time()
    for _ in range(n):
        out = forward_fn(params, video, noise); out.block_until_ready()
    return (time.time() - t0) / n

# ---- forward + backward ----
optimizer = optax.adamw(3e-4)
opt_state = optimizer.init(params)

def loss_fn(params, video, noise, gt_curves, gt_valid, gt_widths, gt_amps, key):
    k1, k2 = jax.random.split(key)
    n1 = sample_batched_noise(k1, BATCH, cfg_u)
    n2 = sample_batched_noise(k2, BATCH, cfg_u)
    p1 = model.apply(params, video, n1, train=True)
    p2 = model.apply(params, video, n2, train=True)
    c1 = decode_curves(p1, cfg_u, pca_mean, pca_basis)
    c2 = decode_curves(p2, cfg_u, pca_mean, pca_basis)
    f1 = unpack_pred(p1); f2 = unpack_pred(p2)
    s1 = jax.nn.sigmoid(f1["score"]); s2 = jax.nn.sigmoid(f2["score"])
    def per(i):
        return compute_energy_loss(
            c1[i], c2[i], s1[i], s2[i],
            f1["width"][i], f2["width"][i], f1["amp"][i], f2["amp"][i],
            gt_curves[i], gt_valid[i], gt_widths[i], gt_amps[i], cfg_e)
    totals, _ = jax.vmap(per)(jnp.arange(BATCH))
    return totals.mean()

@jax.jit
def train_step(params, opt_state, video, noise, gt_curves, gt_valid,
               gt_widths, gt_amps, key):
    loss, grads = jax.value_and_grad(loss_fn)(
        params, video, noise, gt_curves, gt_valid, gt_widths, gt_amps, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Prep dummy data on device
def make_dummy_batch(k):
    outs = sample_sim_batch(jax.random.split(k, BATCH))
    return (outs["clip_median"],
            outs["curves"][:, T // 2],
            outs["flagella"]["alive"],
            outs["flagella"]["width"],
            outs["flagella"]["amp"])

batch_key = jax.random.key(0)
clip, gt_c, gt_v, gt_w, gt_a = make_dummy_batch(batch_key)
n_noise = jnp.zeros((BATCH, H, H, 1))  # placeholder, unused

def train_only(n=10):
    key = jax.random.key(0)
    params_, opt_state_ = params, opt_state
    # Warm
    p_, o_, l = train_step(params_, opt_state_, clip, n_noise, gt_c, gt_v, gt_w, gt_a, key)
    l.block_until_ready()
    t0 = time.time()
    for i in range(n):
        p_, o_, l = train_step(p_, o_, clip, n_noise, gt_c, gt_v, gt_w, gt_a,
                                jax.random.key(i + 1))
        l.block_until_ready()
    return (time.time() - t0) / n

# ---- combined sim + train (like real loop) ----
def full_step(n=10):
    key = jax.random.key(0)
    p_, o_ = params, opt_state
    # Warm
    clip, gt_c, gt_v, gt_w, gt_a = make_dummy_batch(jax.random.key(0))
    p_, o_, l = train_step(p_, o_, clip, n_noise, gt_c, gt_v, gt_w, gt_a, key)
    l.block_until_ready()
    t0 = time.time()
    for i in range(n):
        clip, gt_c, gt_v, gt_w, gt_a = make_dummy_batch(jax.random.key(i))
        p_, o_, l = train_step(p_, o_, clip, n_noise, gt_c, gt_v, gt_w, gt_a,
                                jax.random.key(i + 1))
        l.block_until_ready()
    return (time.time() - t0) / n

print(f"  sim_only          : {sim_only()*1000:.1f} ms / batch = {BATCH/sim_only():.1f} clips/s")
print(f"  fwd_only          : {fwd_only()*1000:.1f} ms / batch = {BATCH/fwd_only():.1f} clips/s")
print(f"  train_step (fixed): {train_only()*1000:.1f} ms / batch = {BATCH/train_only():.1f} clips/s")
print(f"  full (sim+train)  : {full_step()*1000:.1f} ms / batch = {BATCH/full_step():.1f} clips/s")
