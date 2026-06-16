# sim2real_tracking — notes for future Claude sessions

## Project goal
Test whether an object-centric video model can be pretrained (supervised) on a simulator with prior setting A, then adapted (unsupervised reconstruction) to prior setting B, and still produce semantically meaningful latents. The full plan is at `/home/frans/.claude/plans/i-m-starting-a-new-purring-lemon.md`.

## Framework
JAX + Flax (linen). Optax for optim. Orbax for checkpointing. interpax for splines. numpyro is allowed inside simulators only (probabilistic sampling), never inside the model or training loop.

## Conventions

### Latents
- `z_where ∈ R^3` = `(s_raw, tx_raw, ty_raw)` in unconstrained tanh-space. `s = sigmoid(s_raw) ∈ (0,1)`, `tx,ty = tanh(_raw) ∈ (-1,1)`. STN read/write uses these directly.
- `z_pres ∈ [0,1]` — Gumbel-sigmoid with straight-through during train, hard sigmoid at eval.
- `z_what ∈ R^{Zw}` — dual source (MLP head + glimpse skip). Always add, never concat.
- `z_style ∈ R^{Zs}` — one per *video* (not per frame). Modulates decoder via AdaLN.

### Shapes
Public tensors are batch-first with leading `(B, T, ...)`. Model internals scan over T, so inside `model.py` we drop the batch and vmap externally.

### Lessons inherited from cellulose (do NOT undo these without explicit reason)
- **v17 fix**: `head_where = prev_z_where + 0.5 * tanh(linear_zero_init(Q))`. The zero-init of the last layer is load-bearing: it prevents "all cells move in tandem".
- **v18 fix**: dual-source z_what (`mlp_main(Q) + mlp_skip(glimpse_feat)`). Add, not concat. Required for per-slot diversity.
- **v20 fix**: clip composite to [0,1] BEFORE MSE. Otherwise the model exploits cancellation and recon plateaus.
- **v20 fix**: one-sided BCE on z_pres — only penalize false-positives. Let the recon loss pull z_pres up.

### Don't
- Don't import from `/home/frans/cellulose/`. Re-implement cleanly here.
- Don't introduce CellPose or any external seeder. The whole point is that propagation+discovery is model-internal.
- Don't bypass the priors with hard regularizers. Use the `Prior.kl()` API so everything is principled and ablatable.

## Build order
See plan §"Build order". Each task in `TaskList` corresponds to one step.

## Running tests
```bash
cd /home/frans/sim2real_tracking
pip install -e .[dev]
pytest tests/
```

## Known JAX gotchas in this codebase
- Hungarian matching runs on host via `scipy.optimize.linear_sum_assignment`. Wrap result in `jax.lax.stop_gradient` and ferry as int indices.
- `vmap` over the simulators: build batched outputs by `jax.vmap(sim_one)` over key-splits, *not* by sampling inside a Python loop.
- Augmax is used in simulators only. Don't try to put augmax ops inside `jit` boundaries that need to be traced repeatedly.
