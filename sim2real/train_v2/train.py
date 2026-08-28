"""Training loop for DETRSlotModel on sim data (FG-only + real BG patches).

Data pipeline (host, CPU/numpy):
  1. Sample B scenes with `sample_scene(rng, cfg, bg_patch=random_bg)`
  2. Stack clips + energy maps + pack GT
  3. Move to JAX

Training step:
  a. Model forward (jit'd) → outputs
  b. Hungarian match on host using pred + GT
  c. Loss + grad + optim step (jit'd)

No JIT'd Hungarian for now — matching is O(B * S * G) which is trivially fast on CPU
and would require jax.pure_callback to enter JIT (added complexity for negligible speedup).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from sim2real.data import (
    CANONICAL_H, CANONICAL_W, CANONICAL_TARGET_WIDTH_PX,
    FlagellumSimConfig, sample_scene,
)
from sim2real.model_v2 import (
    DETRSlotConfig, DETRSlotModel, compute_loss, pack_gt_batch, run_matches,
)


# ------------------------------- Data --------------------------------

def sample_batch(rng: np.random.Generator, cfg: FlagellumSimConfig, bg_patches: np.ndarray,
                 batch_size: int) -> dict:
    """Sample B scenes and return numpy arrays ready for JAX."""
    scenes = []
    clips = np.empty((batch_size, cfg.T, CANONICAL_H, CANONICAL_W), dtype=np.float32)
    energies = np.empty((batch_size, CANONICAL_H, CANONICAL_W), dtype=np.float32)
    for i in range(batch_size):
        bg = bg_patches[rng.integers(0, len(bg_patches))]
        s = sample_scene(rng, cfg, bg_patch=bg)
        clips[i] = s.clip
        energies[i] = s.energy
        scenes.append(s.latents)
    gt = pack_gt_batch(scenes)
    return dict(clip=clips, energy=energies, **gt)


# ------------------------------- Train state --------------------------------

def create_train_state(rng, cfg: DETRSlotConfig, batch_size: int, T: int, lr: float,
                       warmup_steps: int, total_steps: int, weight_decay: float = 1e-4):
    model = DETRSlotModel(cfg=cfg)
    dummy_clip = jnp.zeros((batch_size, T, CANONICAL_H, CANONICAL_W))
    dummy_energy = jnp.zeros((batch_size, CANONICAL_H, CANONICAL_W))
    params = model.init({"params": rng, "slots": rng}, dummy_clip, dummy_energy)
    # optax subtracts warmup_steps from decay_steps internally, so decay_steps must be > warmup_steps.
    warmup = min(warmup_steps, max(total_steps - 1, 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup,
        decay_steps=max(total_steps, warmup + 1), end_value=lr * 0.01,
    )
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=weight_decay))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optim)
    return state, model, schedule


# ------------------------------- JIT'd steps --------------------------------

def make_train_step(model):
    """Build a jit'd train step. Takes pre-computed Hungarian matches as auxiliary input."""

    GT_KEYS = ("gt_pts", "gt_width", "gt_amp", "gt_polarity", "gt_mask",
               "gt_cell_center", "gt_cell_radius", "gt_cell_amp", "gt_cell_mask")

    def loss_fn(params, batch, match_flag, match_cell, rng):
        out = model.apply(params, batch["clip"], batch["energy"], rngs={"slots": rng})
        gt = {k: batch[k] for k in GT_KEYS}
        losses = compute_loss(out, gt, match_flag, match_cell,
                              class_weight=10.0,
                              pts_weight=1.0, width_weight=0.5,
                              amp_weight=0.5, polarity_weight=0.5,
                              cell_pos_weight=1.0, cell_radius_weight=0.5, cell_amp_weight=0.5)
        return losses["total"], (losses, out)

    @jax.jit
    def train_step(state, batch, match_flag, match_cell, rng):
        (total, (losses, out)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch, match_flag, match_cell, rng)
        state = state.apply_gradients(grads=grads)
        return state, losses, out

    @jax.jit
    def forward_only(params, batch, rng):
        return model.apply(params, batch["clip"], batch["energy"], rngs={"slots": rng})

    return train_step, forward_only


# ------------------------------- Loop --------------------------------

def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading BG patches from {args.bg_patches}")
    bg_data = np.load(args.bg_patches, allow_pickle=True)
    bg_patches = bg_data["patches"]
    print(f"[data] {len(bg_patches)} BG patches available, shape {bg_patches.shape}")

    sim_cfg = FlagellumSimConfig(T=args.T)
    model_cfg = DETRSlotConfig(n_slots=args.n_slots, n_iters=args.n_iters,
                               d_slot=args.d_slot, d_ff_slot=args.d_ff_slot)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    state, model, sched = create_train_state(
        init_key, model_cfg, args.batch_size, args.T, args.lr,
        warmup_steps=args.warmup, total_steps=args.steps,
    )
    print(f"[model] params: {sum(x.size for x in jax.tree_util.tree_leaves(state.params))/1e6:.2f}M")

    train_step, forward_only = make_train_step(model)

    rng_np = np.random.default_rng(args.seed)
    t0 = time.time()
    log = []
    for step in range(1, args.steps + 1):
        batch_np = sample_batch(rng_np, sim_cfg, bg_patches, args.batch_size)

        # Compute both matches on host using a forward-only call
        key, sub = jax.random.split(key)
        batch_jax = {k: jnp.asarray(v) for k, v in batch_np.items()}
        out = forward_only(state.params, batch_jax, sub)
        out_np = {k: np.asarray(v) for k, v in out.items()}
        match_flag_np, match_cell_np = run_matches(out_np, batch_np)
        # Full train step with matches as aux
        key, sub2 = jax.random.split(key)
        state, losses, _ = train_step(
            state, batch_jax,
            jnp.asarray(match_flag_np), jnp.asarray(match_cell_np), sub2,
        )

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            eta = (args.steps - step) / max(steps_per_sec, 1e-6)
            entry = dict(
                step=step,
                total=float(losses["total"]),
                class_=float(losses["class_"]),
                pts=float(losses["pts"]),
                width=float(losses["width"]),
                amp=float(losses["amp"]),
                polarity=float(losses["polarity"]),
                cell_pos=float(losses["cell_pos"]),
                cell_rad=float(losses["cell_rad"]),
                cell_amp=float(losses["cell_amp"]),
                n_matched_flag=float(losses["n_matched_flag"]),
                n_matched_cell=float(losses["n_matched_cell"]),
            )
            log.append(entry)
            print(f"  step {step:>5}/{args.steps}  total={entry['total']:.2f}  "
                  f"cls={entry['class_']:.2f}  pts={entry['pts']:.1f}  "
                  f"cell_pos={entry['cell_pos']:.1f} rad={entry['cell_rad']:.1f}  "
                  f"nf={entry['n_matched_flag']:.0f} nc={entry['n_matched_cell']:.0f}  "
                  f"({steps_per_sec:.1f} sps, eta={eta:.0f}s)")

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = out_dir / f"ckpt_step_{step}.pkl"
            _save_ckpt(state, model_cfg, sim_cfg, ckpt_path)
            with (out_dir / "loss_log.json").open("w") as f:
                json.dump(log, f, indent=2)

    print(f"[done] {time.time()-t0:.0f}s total")


def _save_ckpt(state, model_cfg: DETRSlotConfig, sim_cfg: FlagellumSimConfig, path: Path):
    import pickle
    params = jax.device_get(state.params)
    with open(path, "wb") as f:
        pickle.dump({"params": params, "model_cfg": model_cfg, "sim_cfg": sim_cfg,
                     "step": int(state.step)}, f)
    print(f"[save] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--n-slots", type=int, default=8)
    ap.add_argument("--n-iters", type=int, default=3)
    ap.add_argument("--d-slot", type=int, default=128)
    ap.add_argument("--d-ff-slot", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg-patches", default="/home/frans/sim2real_tracking/data_cache/bg_patches_v0.npz")
    ap.add_argument("--out-dir", default="/home/frans/sim2real_tracking/runs/detr_slot/v0")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
