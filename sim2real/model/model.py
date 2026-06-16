"""Full SlotVideoModel — encode → slot transformer → heads → decode → composite, scanned over T.

The model operates on **one video** (no batch dim); the trainer vmaps over the batch externally.

Forward signature:
    apply(params, video, key) -> ModelOut

`video` has shape (T, H, W, C). Returns a `ModelOut` with shapes documented in `sim2real.types`.

Notes:
- Slot identity is the slot index. The GRU state (`slot_h`) is carried across frames.
- z_where prediction is residual on top of the previous frame's z_where.
- z_pres is Gumbel-sigmoid with straight-through.
- Discovery is implemented as a second cross-attention pass with a residual mask (1 - Σ alive_mask).
  Discovery is wired here so the discover-pass tokens drive z_pres for *dormant* slots; for alive
  slots, propagate-pass tokens drive everything. The split is via z_pres_prev gating.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from sim2real.model.encoder import FrameEncoder
from sim2real.model.glimpse import GlimpseDecoder, GlimpseEncoder, SegHead
from sim2real.model.heads import PresHead, WhatHead, WhereHead
from sim2real.model.slot_transformer import SlotTransformer
from sim2real.model.stn import stn_read, stn_write
from sim2real.model.style import StyleEncoder
from sim2real.model.temporal import SlotGRU
from sim2real.types import ModelOut

Array = jnp.ndarray


@dataclass(frozen=True)
class ModelConfig:
    n_max: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_transformer_layers: int = 2
    z_what_dim: int = 64
    z_style_dim: int = 16
    glimpse_size: int = 16
    n_vit_layers: int = 1
    stem_channels: tuple = (16, 32, 64)
    where_scale: float = 0.5
    pres_init_bias: float = -1.0
    pres_tau: float = 0.5
    use_discovery: bool = True


class SlotVideoModel(nn.Module):
    cfg: ModelConfig

    def setup(self):
        c = self.cfg
        self.encoder = FrameEncoder(
            d_model=c.d_model, n_vit_layers=c.n_vit_layers, stem_channels=tuple(c.stem_channels)
        )
        self.slot_transformer = SlotTransformer(
            n_max=c.n_max, d_model=c.d_model, n_heads=c.n_heads, n_layers=c.n_transformer_layers
        )
        self.where_head = WhereHead(scale=c.where_scale, hidden=c.d_model)
        self.pres_head = PresHead(hidden=c.d_model, init_bias=c.pres_init_bias)
        self.what_head = WhatHead(z_what_dim=c.z_what_dim, hidden=c.d_model)
        self.glimpse_encoder = GlimpseEncoder(feat_dim=c.z_what_dim, channels=(16, 32))
        self.glimpse_decoder = GlimpseDecoder(
            glimpse_size=c.glimpse_size, z_what_dim=c.z_what_dim, channels=(32, 16)
        )
        self.seg_head = SegHead(glimpse_size=c.glimpse_size, hidden=c.d_model)
        self.slot_gru = SlotGRU(d_model=c.d_model)
        self.style_encoder = StyleEncoder(z_style_dim=c.z_style_dim, hidden=c.d_model)

        # Learned per-slot z_where init (used on frame 0 as the residual anchor).
        # Initialize positions roughly spread across the canvas via small noise.
        self.z_where_init = self.param(
            "z_where_init",
            lambda key: jax.random.normal(key, (c.n_max, 3)) * 0.1,
        )

    def _per_slot_head(self, q, key_p, key_w, prev_zwhere, image):
        """Apply heads + decoder + STN for one slot. q: (d,) ; prev_zwhere: (3,)."""
        cfg = self.cfg
        zwhere = self.where_head(q, prev_zwhere)
        glimpse = stn_read(image, zwhere, cfg.glimpse_size)
        glimpse_feat = self.glimpse_encoder(glimpse)
        zpres, zpres_logit = self.pres_head(q, key_p, tau=cfg.pres_tau, straight_through=True)
        zwhat, mu_w, lv_w = self.what_head(q, glimpse_feat, key_w)
        appear_patch, mask_logit_patch = self.glimpse_decoder(zwhat)
        # Separate seg head (independent from appearance mask)
        seg_logit_patch = self.seg_head(zwhat, zwhere)
        # Place onto canvas
        appear_canvas = stn_write(appear_patch, zwhere, image.shape[0])               # (R, R, 1)
        mask_appear_canvas = stn_write(
            nn.sigmoid(mask_logit_patch) * zpres, zwhere, image.shape[0]
        )[..., 0]                                                                     # (R, R)
        mask_seg_canvas = stn_write(nn.sigmoid(seg_logit_patch), zwhere, image.shape[0])[..., 0]
        return zwhere, zpres, zpres_logit, zwhat, mu_w, lv_w, glimpse_feat, appear_canvas, mask_appear_canvas, mask_seg_canvas

    def __call__(self, video: Array, key):
        cfg = self.cfg
        T, H, W, C = video.shape

        # 1) Encode every frame.
        feats, pools = jax.vmap(self.encoder)(video)                                  # (T, h, w, d), (T, d)
        h_prime, w_prime = feats.shape[1], feats.shape[2]

        # 2) z_style from pooled features.
        k_style = jax.random.fold_in(key, 0)
        z_style_sample, z_style_mu, z_style_logvar = self.style_encoder(pools, k_style)

        # 3) Scan over time.
        slot_h0 = jnp.zeros((cfg.n_max, cfg.d_model))
        prev_zwhere0 = self.z_where_init                                              # (N, 3)
        prev_zpres0 = jnp.zeros((cfg.n_max,))                                         # all dormant
        prev_zwhat0 = jnp.zeros((cfg.n_max, cfg.z_what_dim))

        def step(carry, inputs):
            slot_h, prev_zwhere, prev_zpres, prev_zwhat = carry
            t_idx, feat_grid, image, k = inputs
            del t_idx

            # Residual mask for discovery: 1 - Σ_slots prev_zpres * indicator(in-glimpse-region).
            # We approximate this by pooling the previous-frame predicted masks; since we don't have
            # them in carry, we use a soft "alive presence" → low-res indicator from prev_zwhere.
            if cfg.use_discovery:
                # Build a coarse alive-mask at encoder resolution from prev_zwhere using gaussian bumps.
                grid_y = (jnp.linspace(-1.0, 1.0, h_prime))
                grid_x = (jnp.linspace(-1.0, 1.0, w_prime))
                yy, xx = jnp.meshgrid(grid_y, grid_x, indexing="ij")
                s = jax.nn.sigmoid(prev_zwhere[:, 0])
                tx = jnp.tanh(prev_zwhere[:, 1])
                ty = jnp.tanh(prev_zwhere[:, 2])
                d2 = (xx[None] - tx[:, None, None]) ** 2 + (yy[None] - ty[:, None, None]) ** 2
                alive_bumps = jnp.exp(-0.5 * d2 / (s[:, None, None] ** 2 + 1e-4))
                explained = jnp.clip(
                    jnp.sum(prev_zpres[:, None, None] * alive_bumps, axis=0), 0.0, 1.0
                )
                residual_mask_pixel = 1.0 - explained
            else:
                residual_mask_pixel = None

            q_prop, q_disc, _ = self.slot_transformer(feat_grid, slot_h, residual_mask_pixel)

            # Choose tokens per slot: alive (prev_zpres ≈ 1) → q_prop; dormant → q_disc.
            alive = prev_zpres[:, None]                                              # (N, 1)
            q = alive * q_prop + (1.0 - alive) * q_disc                              # (N, d)

            # Heads + decoder per slot.
            k_p = jax.random.fold_in(k, 1)
            k_w = jax.random.fold_in(k, 2)
            keys_p = jax.random.split(k_p, cfg.n_max)
            keys_w = jax.random.split(k_w, cfg.n_max)

            (zwhere, zpres, zpres_logit, zwhat, mu_w, lv_w, glimpse_feat,
             appear_canvas, mask_appear_canvas, mask_seg_canvas) = jax.vmap(
                self._per_slot_head, in_axes=(0, 0, 0, 0, None)
            )(q, keys_p, keys_w, prev_zwhere, image)

            # Composite frame from alive slots only (z_pres already gates the appearance mask).
            num = jnp.sum(appear_canvas * mask_appear_canvas[..., None], axis=0)
            den = jnp.sum(mask_appear_canvas, axis=0)[..., None] + 1e-6
            composite = jnp.clip(num / den, 0.0, 1.0)                                # (H, W, 1)

            # Slot state update (vmapped GRU).
            gru_input = jnp.concatenate(
                [q, zwhere, zwhat, zpres[:, None]], axis=-1
            )                                                                         # (N, d + 3 + Zw + 1)
            new_h = jax.vmap(self.slot_gru, in_axes=(0, 0))(slot_h, gru_input)

            new_carry = (new_h, zwhere, zpres, zwhat)
            out = dict(
                z_where=zwhere, z_pres=zpres, z_pres_logit=zpres_logit,
                z_what=zwhat, mu_w=mu_w, lv_w=lv_w,
                glimpse_feat=glimpse_feat,
                appear_pred=appear_canvas,
                mask_appear_pred=mask_appear_canvas,
                mask_seg_pred=mask_seg_canvas,
                composite=composite,
            )
            return new_carry, out

        # NOTE: We use a Python-unrolled loop rather than `jax.lax.scan` because the slot
        # transformer's parameters are created inside `setup()` and we need linen's parameter
        # tracking, which doesn't compose with `jax.lax.scan` without `nn.scan`. For T ≤ ~30
        # the unrolled graph is small enough not to matter.
        keys_t = jax.random.split(jax.random.fold_in(key, 1), T)
        carry = (slot_h0, prev_zwhere0, prev_zpres0, prev_zwhat0)
        outs = []
        for t in range(T):
            carry, out_t = step(carry, (jnp.asarray(t), feats[t], video[t], keys_t[t]))
            outs.append(out_t)

        traj = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *outs)

        return ModelOut(
            z_where=traj["z_where"],                                                  # (T, N, 3)
            z_pres=traj["z_pres"],                                                    # (T, N)
            z_what=traj["z_what"],                                                    # (T, N, Zw)
            masks_pred=traj["mask_seg_pred"],                                         # (T, N, H, W)
            appear_pred=traj["appear_pred"][..., 0],                                  # (T, N, H, W)
            composite=traj["composite"],                                              # (T, H, W, 1)
            z_style=z_style_sample,                                                   # (Zs,)
            aux=dict(
                z_style_mu=z_style_mu,
                z_style_logvar=z_style_logvar,
                z_pres_logit=traj["z_pres_logit"],
                mu_w=traj["mu_w"],
                lv_w=traj["lv_w"],
                mask_appear_pred=traj["mask_appear_pred"],
            ),
        )
