"""Public pytree types shared across the project.

Conventions:
- `z_where = (s_raw, tx_raw, ty_raw)` in unconstrained tanh-space. The STN reads `s = sigmoid(s_raw)`,
  `tx,ty = tanh(*_raw)`.
- Batch tensors are leading-axis `(B, T, ...)` outside the model. The model scans over T internally
  and is wrapped in `vmap` over B by the trainer.
"""

from __future__ import annotations

from typing import Any, Optional

import flax.struct
import jax.numpy as jnp

Array = jnp.ndarray


@flax.struct.dataclass
class SimSample:
    """One batch sampled from a simulator.

    Shapes (leading B, T):
      video:   (B, T, H, W, C)        C=1 grayscale; values in [0,1].
      z_where: (B, T, N, 3)           ground-truth latents in tanh-space.
      z_pres:  (B, T, N)              in {0,1}.
      z_what:  (B, N, Zw) | None      simulator-defined ground-truth appearance code (optional).
      z_style: (B, Zs)                one per video.
      masks:   (B, T, N, H, W)        soft masks in [0,1]; the seg-supervision target.
      meta:    dict                   sim_kind, per-sim diagnostics; not a traced pytree.
    """

    video: Array
    z_where: Array
    z_pres: Array
    z_style: Array
    masks: Array
    z_what: Optional[Array] = None
    meta: dict[str, Any] = flax.struct.field(pytree_node=False, default_factory=dict)


@flax.struct.dataclass
class ModelOut:
    """One forward pass of `SlotVideoModel`.

    Per single video (the trainer vmaps over batch):
      z_where:    (T, N_max, 3)
      z_pres:     (T, N_max)
      z_what:     (T, N_max, Zw)
      masks_pred: (T, N_max, H, W)        from the dedicated seg head
      appear_pred:(T, N_max, H, W)        appearance × mask used in composite
      composite:  (T, H, W, C)            clipped to [0,1]
      z_style:    (Zs,)                   one per video
      aux:        dict                    discovery_scores, glimpse_feat, mu_video, prop_h ...
    """

    z_where: Array
    z_pres: Array
    z_what: Array
    masks_pred: Array
    appear_pred: Array
    composite: Array
    z_style: Array
    aux: dict[str, Any] = flax.struct.field(default_factory=dict)


@flax.struct.dataclass
class StyleLatent:
    mu: Array
    log_sigma: Array
    sample: Array
