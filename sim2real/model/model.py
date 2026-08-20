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

from sim2real.model.background import BackgroundRenderer
from sim2real.model.encoder import FrameEncoder
from sim2real.model.glimpse import GlimpseDecoder, GlimpseEncoder, GroupedDecoder
from sim2real.model.heads import PresHead, WhatHead, WhereHead
from sim2real.model.isa import ISAStack
from sim2real.model.neural_em import NeuralEMStack
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
    stem_strides: tuple = (2, 2, 2)
    where_scale: float = 0.5
    pres_init_bias: float = -1.0
    pres_tau: float = 0.5
    use_discovery: bool = True
    # When True, the recon path uses stop_gradient on z_where (render position) and z_pres
    # (presence gate) before stn_write. Decoder + z_what are trained by L_recon; the pose head
    # is trained exclusively by L_where, the pres head by L_pres, the seg head by L_mask.
    # See cellulose/sqair_cells/train_mixed.py:662-663 for the original pattern.
    stop_grad_recon_path: bool = True
    # When True, ALSO stop gradient on z_what along the composite recon path — decoder is called
    # twice: once with gradient (for L_appear_glimpse / L_mask_glimpse), once with stop_grad(zwhat)
    # (for the composite). Cleanly decouples recognition from generation: z_what is trained only
    # by supervised losses, and the composite recon loss trains only decoder weights, not the
    # slot-transformer that produced z_what.
    stop_grad_recon_zwhat: bool = False
    # PresHead capacity + optional cross-attention to image features.
    pres_hidden: int = 128           # was hard-coded to d_model. Bump to 256 to fatten pres head.
    pres_depth: int = 1              # number of hidden MLP layers before the logit projection.
    pres_image_attn: bool = False    # if True, PresHead cross-attends to encoder feature grid.
    # Slot Attention-style softmax-OVER-SLOTS in the transformer's cross-attention layers.
    # Turns DETR-style attention (each slot independently attends to keys) into Slot Attention
    # competition (each key claimed by one slot), targeting the "duplicate slots on same object"
    # failure mode where our current standard cross-attention lets two slots collapse to one cell.
    slot_competing_cross: bool = False
    # Anchor-based slot queries: replace learnable per-slot embeddings with a fixed grid of
    # anchor positions encoded via sinusoidal features. Ties slot identity to spatial location,
    # avoiding the permutation-averaged-gradient collapse of learnable queries.
    anchor_slots: bool = False
    # Proper iterative Slot Attention (Locatello 2020) as replacement for the DETR-style
    # slot transformer. Uses softmax-over-slots + GRU update + n_transformer_layers iterations
    # of shared-weight refinement. Turns the model into SAVi-with-our-heads.
    use_slot_attention: bool = False
    # Neural EM: iterate structured latents (z_where, z_pres, z_what) directly. Replaces
    # the slot_transformer + heads with NeuralEMStack. Slot state IS the structured
    # object parameters — no opaque hidden vector. See sim2real/model/neural_em.py.
    use_neural_em: bool = False
    # Invariant Slot Attention (Biza et al 2302.04973). Extends Neural EM with per-slot
    # equivariant position encoding — the attention key sees pixel offsets in the SLOT's frame
    # (rotated + scaled), so a translated / rotated / scaled object is picked up by the same
    # slot regardless of image location. See sim2real/model/isa.py.
    use_isa: bool = False
    # ISA-specific knobs (ignored when use_isa=False).
    #   isa_abs_pe_weight: > 0 mixes absolute PE into K_base (fallback signal when
    #                      equivariant PE fails from bad slot pose init).
    #   isa_anchor_init:   initialize z_where_init to a fixed grid of slot positions
    #                      (breaks the chicken-and-egg between attention & pose).
    isa_abs_pe_weight: float = 0.0
    isa_anchor_init: bool = False
    # Anchor init for NEM path (analogous to isa_anchor_init but always fixed, not learned).
    #   anchor_init_fixed: use a fixed grid of slot positions as z_where_init, NOT a learned
    #                      param. Provides consistent cross-video symmetry breaking.
    #   z_what_init_std:   std of the learned z_what_init random init. Wider = more per-slot
    #                      identity discrimination at frame 0.
    #   nem_attn_temp:     softmax temperature in NEM E-step. <1 = sharper responsibility.
    #                      Addresses "slot collapse to means" where slots drift between cells.
    anchor_init_fixed: bool = False
    z_what_init_std: float = 0.2
    nem_attn_temp: float = 1.0
    # DETR-style bg slot: phantom "null slot" added to softmax competition. Prevents real
    # slots from becoming background sinks (which contaminates their centroid).
    nem_use_bg_slot: bool = False
    # When True, a per-video background field is rendered from z_style and slots composite OVER
    # it. Removes the L_recon = 0.21 floor caused by "unexplained pixels → composite=0".
    use_background: bool = True
    bg_base_res: int = 8
    bg_channels: tuple = (64, 32, 16)
    # K-way hierarchical group latent on z_what (enumerated, no sampling). n_groups = 1 means
    # the vanilla decoder is used (no grouping); n_groups > 1 wraps the decoder in
    # GroupedDecoder which renders K candidates and marginalizes.
    n_groups: int = 1


class SlotVideoModel(nn.Module):
    cfg: ModelConfig

    def setup(self):
        c = self.cfg
        self.encoder = FrameEncoder(
            d_model=c.d_model, n_vit_layers=c.n_vit_layers,
            stem_channels=tuple(c.stem_channels),
            stem_strides=tuple(c.stem_strides),
        )
        self.slot_transformer = SlotTransformer(
            n_max=c.n_max, d_model=c.d_model, n_heads=c.n_heads, n_layers=c.n_transformer_layers,
            slot_competing_cross=c.slot_competing_cross,
            anchor_slots=c.anchor_slots,
            use_slot_attention=c.use_slot_attention,
        )
        self.where_head = WhereHead(scale=c.where_scale, hidden=c.d_model)
        self.pres_head = PresHead(
            hidden=c.pres_hidden, depth=c.pres_depth, init_bias=c.pres_init_bias,
            image_attn=c.pres_image_attn,
        )
        self.what_head = WhatHead(z_what_dim=c.z_what_dim, hidden=c.d_model)
        self.glimpse_encoder = GlimpseEncoder(feat_dim=c.z_what_dim, channels=(16, 32))
        if c.n_groups > 1:
            self.glimpse_decoder = GroupedDecoder(
                n_groups=c.n_groups,
                glimpse_size=c.glimpse_size,
                z_what_dim=c.z_what_dim,
                channels=(32, 16),
            )
        else:
            self.glimpse_decoder = GlimpseDecoder(
                glimpse_size=c.glimpse_size, z_what_dim=c.z_what_dim, channels=(32, 16)
            )
        # NOTE: removed standalone SegHead — masks_pred now reuses the GlimpseDecoder's
        # mask_logit channel. Saves params and lets the L_mask gradient train the same network
        # as L_recon, instead of an independent head that empirically didn't converge.
        self.slot_gru = SlotGRU(d_model=c.d_model)
        self.style_encoder = StyleEncoder(z_style_dim=c.z_style_dim, hidden=c.d_model)
        if c.use_background:
            # out_res set at __call__ time would require dynamic shapes; we instead use a
            # placeholder out_res = 128 which is then cropped to the actual frame size if needed.
            self.bg_renderer = BackgroundRenderer(
                out_res=128,
                base_res=c.bg_base_res,
                channels=tuple(c.bg_channels),
                z_style_dim=c.z_style_dim,
            )

        # Learned per-slot z_where init (used on frame 0 as the residual anchor).
        # 5-dim affine: (sx_raw, sy_raw, theta_raw, tx_raw, ty_raw).
        # Scale channels biased toward sigmoid ~0.10 (logit -2.2) — the empirical mean of
        # many_cells GT scale. Without this, sigmoid(N(0, 0.1)) ≈ 0.50 (5× too big) and the
        # mask decoder trains to output a mostly-empty patch inside the too-large region.
        # Translation channels spread (stddev 0.5) so slots start across the image.
        # With isa_anchor_init: replace random pos with a fixed grid so ISA has a good bootstrap
        # pose (avoids attention <-> pose chicken-and-egg failure).
        # With anchor_init_fixed: skip the learned param entirely — z_where_init is computed
        # inline as a fixed grid at forward time (see __call__).
        if not c.anchor_init_fixed:
            def _zwhere_init(key):
                k_scale, k_theta, k_pos = jax.random.split(key, 3)
                scale = -2.2 + jax.random.normal(k_scale, (c.n_max, 2)) * 0.3   # sigmoid ~ 0.10
                theta = jax.random.normal(k_theta, (c.n_max, 1)) * 0.1
                if c.isa_anchor_init:
                    # Grid of anchor positions covering [-0.7, 0.7]^2. Tile to n_max.
                    import math
                    side = int(math.ceil(math.sqrt(c.n_max)))
                    lin = jnp.linspace(-0.7, 0.7, side)
                    gy, gx = jnp.meshgrid(lin, lin, indexing="ij")
                    pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)[:c.n_max]        # (n_max, 2)
                    pos = jnp.arctanh(jnp.clip(pts, -0.98, 0.98))                              # (n_max, 2)
                else:
                    pos = jax.random.normal(k_pos, (c.n_max, 2)) * 0.5
                return jnp.concatenate([scale, theta, pos], axis=-1)
            self.z_where_init = self.param("z_where_init", _zwhere_init)

        if c.use_neural_em:
            # NeuralEMStack runs n_transformer_layers iterations of E+M steps.
            self.neural_em = NeuralEMStack(
                n_max=c.n_max, d_model=c.d_model, z_what_dim=c.z_what_dim,
                n_iters=c.n_transformer_layers, d_pos=32,
                attn_temp=c.nem_attn_temp,
                use_bg_slot=c.nem_use_bg_slot,
            )
            # Per-slot initial z_what — small learned tensor, one row per slot. Provides the
            # symmetry-breaking bias for slot identity at t=0 (analogous to slot_init_bias in
            # Slot Attention).
            self.z_what_init = self.param(
                "z_what_init",
                nn.initializers.normal(stddev=c.z_what_init_std),
                (c.n_max, c.z_what_dim),
            )

        if c.use_isa:
            self.isa = ISAStack(
                n_max=c.n_max, d_model=c.d_model, z_what_dim=c.z_what_dim,
                n_iters=c.n_transformer_layers, d_pos=32,
                abs_pe_weight=c.isa_abs_pe_weight,
            )
            self.z_what_init_isa = self.param(
                "z_what_init_isa",
                nn.initializers.normal(stddev=c.z_what_init_std),
                (c.n_max, c.z_what_dim),
            )

    def _predict_zwhere(self, q, prev_zwhere):
        return self.where_head(q, prev_zwhere)

    def _per_slot_decode_only(self, zwhat, zwhere, zpres, image):
        """Decoder + STN + per-slot canvases for the Neural EM path.

        Skips all head calls (pres_head/what_head/where_head) — latents are supplied by
        the Neural EM refiner. Returns per-slot canvases used by the composite.
        """
        cfg = self.cfg
        glimpse = stn_read(image, zwhere, cfg.glimpse_size)
        glimpse_feat = self.glimpse_encoder(glimpse)

        if cfg.n_groups > 1:
            raise NotImplementedError("n_groups>1 not supported with use_neural_em")
        appear_patch, mask_logit_patch = self.glimpse_decoder(zwhat)                    # (g, g, 1) each

        mask_prob_patch = nn.sigmoid(mask_logit_patch)                                  # (g, g, 1)

        if cfg.stop_grad_recon_path:
            recon_zwhere = jax.lax.stop_gradient(zwhere)
            recon_zpres = jax.lax.stop_gradient(zpres)
        else:
            recon_zwhere = zwhere
            recon_zpres = zpres

        appear_canvas = stn_write(appear_patch, recon_zwhere, image.shape[0])           # (R, R, 1)
        mask_appear_canvas = stn_write(
            mask_prob_patch * recon_zpres, recon_zwhere, image.shape[0]
        )[..., 0]                                                                        # (R, R)
        mask_seg_canvas = stn_write(mask_prob_patch, zwhere, image.shape[0])[..., 0]    # (R, R)

        return (appear_canvas, mask_appear_canvas, mask_seg_canvas,
                mask_logit_patch, appear_patch, glimpse_feat)

    def _per_slot_head(self, q, key_p, key_w, zwhere_pred, render_zwhere, render_zpres, image, feat_grid):
        """Apply heads + decoder + STN for one slot.

        Args:
          q: (d,) refined slot token.
          zwhere_pred: (3,) the model's PREDICTED z_where — kept in outputs so L_where can train it.
          render_zwhere: (3,) the z_where used by `stn_read` / `stn_write`. Equals
            `zwhere_pred` in self-prediction mode, or GT z_where in teacher-forcing mode.
          render_zpres: () — the scalar presence used to gate the appearance composite. Equals the
            predicted z_pres in self-prediction mode, or GT z_pres in teacher-forcing mode.
          image: (H, W, C).
        """
        cfg = self.cfg
        glimpse = stn_read(image, render_zwhere, cfg.glimpse_size)
        glimpse_feat = self.glimpse_encoder(glimpse)
        zpres, zpres_logit = self.pres_head(
            q, key_p, tau=cfg.pres_tau, straight_through=True,
            image_feats=feat_grid if cfg.pres_image_attn else None,
        )
        zwhat, mu_w, lv_w = self.what_head(q, glimpse_feat, key_w)
        if self.cfg.n_groups > 1:
            appear_patch, mask_logit_patch, g_post = self.glimpse_decoder(zwhat, q)
        else:
            appear_patch, mask_logit_patch = self.glimpse_decoder(zwhat)
            g_post = jnp.ones((1,))

        # Optional second decoder call with stop_grad(zwhat), used only for the composite recon.
        # Decoder params still get gradient from this path (they're leaves); zwhat does not.
        if cfg.stop_grad_recon_zwhat:
            zwhat_recon = jax.lax.stop_gradient(zwhat)
            if self.cfg.n_groups > 1:
                appear_patch_recon, mask_logit_patch_recon, _ = self.glimpse_decoder(zwhat_recon, q)
            else:
                appear_patch_recon, mask_logit_patch_recon = self.glimpse_decoder(zwhat_recon)
        else:
            appear_patch_recon = appear_patch
            mask_logit_patch_recon = mask_logit_patch

        if cfg.stop_grad_recon_path:
            recon_zwhere = jax.lax.stop_gradient(render_zwhere)
            recon_zpres = jax.lax.stop_gradient(render_zpres)
        else:
            recon_zwhere = render_zwhere
            recon_zpres = render_zpres

        # The mask_logit_patch from the GlimpseDecoder is reused for BOTH the recon composite
        # gating AND the supervised L_mask target (overfit-one diagnostic showed an independent
        # SegHead can't compete with the recon-driven gradient on the decoder's mask channel).
        # Recon-side: gate by recon_zpres (stop_grad if configured).
        # Seg-side: same mask, no zpres gate, full gradient through stn_write.
        mask_prob_patch = nn.sigmoid(mask_logit_patch)                                  # (g, g, 1)
        mask_prob_patch_recon = nn.sigmoid(mask_logit_patch_recon)

        appear_canvas = stn_write(appear_patch_recon, recon_zwhere, image.shape[0])     # (R, R, 1)
        mask_appear_canvas = stn_write(
            mask_prob_patch_recon * recon_zpres, recon_zwhere, image.shape[0]
        )[..., 0]                                                                       # (R, R)
        # mask_seg_canvas keeps the original (non-stopped) render_zwhere so L_mask's gradient
        # can flow into pose if stop_grad_recon_path is False later; today render_zwhere is
        # already a teacher constant when teacher forcing is on.
        mask_seg_canvas = stn_write(mask_prob_patch, render_zwhere, image.shape[0])[..., 0]
        return (
            zwhere_pred, zpres, zpres_logit, zwhat, mu_w, lv_w, glimpse_feat,
            appear_canvas, mask_appear_canvas, mask_seg_canvas, g_post,
            mask_logit_patch, appear_patch,
        )

    def __call__(self, video: Array, key, *, teacher_zwhere=None, teacher_zpres=None,
                 bootstrap_zwhere0=None):
        """Forward one video.

        Args:
          video: (T, H, W, C).
          key:   PRNG key.
          teacher_zwhere: optional (T, N, 3) — GT z_where in slot-index order. If provided, the
            residual anchor for `head_where` becomes `teacher_zwhere[t-1]` and `stn_read/write`
            both use `teacher_zwhere[t]`. The predicted z_where is still produced and still
            trained via `L_where`.
          teacher_zpres: optional (T, N) — GT z_pres in slot-index order. If provided, the
            alive/dormant gate that selects between `q_prop` and `q_disc` uses
            `teacher_zpres[t-1]` instead of the model's previous prediction.
        """
        cfg = self.cfg
        T, H, W, C = video.shape

        # 1) Encode every frame.
        feats, pools = jax.vmap(self.encoder)(video)                                  # (T, h, w, d), (T, d)
        h_prime, w_prime = feats.shape[1], feats.shape[2]

        # 2) z_style from pooled features.
        k_style = jax.random.fold_in(key, 0)
        z_style_sample, z_style_mu, z_style_logvar = self.style_encoder(pools, k_style)

        # 2b) Per-video background field — same across frames.
        if cfg.use_background:
            bg_frame = self.bg_renderer(z_style_sample)                              # (128, 128, 1)
            # Crop to actual frame size if smaller. (Sim runs at 128 today, so no-op.)
            bg_frame = bg_frame[:H, :W]
        else:
            bg_frame = jnp.zeros((H, W, 1))

        # 3) Scan over time.
        slot_h0 = jnp.zeros((cfg.n_max, cfg.d_model))
        # Frame-0 residual anchor. Precedence:
        #   1. bootstrap_zwhere0 (SAVi-style GT bootstrap) if provided
        #   2. Fixed anchor grid if `anchor_init_fixed` is set (NOT a learned param)
        #   3. Learned self.z_where_init
        if bootstrap_zwhere0 is not None:
            prev_zwhere0 = bootstrap_zwhere0
        elif cfg.anchor_init_fixed:
            import math
            side = int(math.ceil(math.sqrt(cfg.n_max)))
            lin = jnp.linspace(-0.7, 0.7, side)
            gy, gx = jnp.meshgrid(lin, lin, indexing="ij")
            pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)[:cfg.n_max]
            pos_raw = jnp.arctanh(jnp.clip(pts, -0.98, 0.98))
            scale_raw = jnp.full((cfg.n_max, 2), -2.2)                                 # sigmoid ~0.10
            theta_raw = jnp.zeros((cfg.n_max, 1))
            prev_zwhere0 = jnp.concatenate([scale_raw, theta_raw, pos_raw], axis=-1)   # (N, 5)
        else:
            prev_zwhere0 = self.z_where_init                                            # (N, 5)
        prev_zpres0 = jnp.zeros((cfg.n_max,))                                         # all dormant
        prev_zwhat0 = jnp.zeros((cfg.n_max, cfg.z_what_dim))

        def step(carry, inputs, prev_zwhere_anchor, prev_zpres_gate,
                 render_zwhere_teacher, render_zpres_teacher):
            slot_h, _prev_zwhere_carry, _prev_zpres_carry, prev_zwhat = carry
            feat_grid, image, k = inputs

            # Use teacher anchor/gate when given, otherwise the predicted carry values.
            prev_zwhere = (
                prev_zwhere_anchor if prev_zwhere_anchor is not None else _prev_zwhere_carry
            )
            prev_zpres = (
                prev_zpres_gate if prev_zpres_gate is not None else _prev_zpres_carry
            )

            # Residual mask for discovery: 1 - Σ_slots prev_zpres * indicator(in-glimpse-region).
            if cfg.use_discovery:
                grid_y = (jnp.linspace(-1.0, 1.0, h_prime))
                grid_x = (jnp.linspace(-1.0, 1.0, w_prime))
                yy, xx = jnp.meshgrid(grid_y, grid_x, indexing="ij")
                # Use max(sx, sy) for the discovery bump radius — anisotropic slots cover at
                # most this radius from their center.
                sx = jax.nn.sigmoid(prev_zwhere[:, 0])
                sy = jax.nn.sigmoid(prev_zwhere[:, 1])
                s = jnp.maximum(sx, sy)
                tx = jnp.tanh(prev_zwhere[:, 3])
                ty = jnp.tanh(prev_zwhere[:, 4])
                d2 = (xx[None] - tx[:, None, None]) ** 2 + (yy[None] - ty[:, None, None]) ** 2
                alive_bumps = jnp.exp(-0.5 * d2 / (s[:, None, None] ** 2 + 1e-4))
                explained = jnp.clip(
                    jnp.sum(prev_zpres[:, None, None] * alive_bumps, axis=0), 0.0, 1.0
                )
                residual_mask_pixel = 1.0 - explained
            else:
                residual_mask_pixel = None

            q_prop, q_disc, _, q_prop_layers, q_disc_layers = self.slot_transformer(
                feat_grid, slot_h, residual_mask_pixel
            )

            # Choose tokens per slot: alive → q_prop; dormant → q_disc.
            alive = prev_zpres[:, None]
            q = alive * q_prop + (1.0 - alive) * q_disc

            # DETR-style deep supervision: at each *intermediate* discover layer (i.e. all
            # except the last, which is the main q), run the where + pres heads using the
            # SAME head parameters, to be supervised by the loss. We mix prop and disc per
            # layer the same way q is mixed for the main prediction.
            aux_zwhere_layers = []
            aux_zpres_logit_layers = []
            if len(q_disc_layers) > 1:
                k_dummy = jax.random.fold_in(k, 99)
                k_dummy_per_slot = jax.random.split(k_dummy, cfg.n_max)
                for li in range(len(q_disc_layers) - 1):
                    q_li = alive * q_prop_layers[li] + (1.0 - alive) * q_disc_layers[li]
                    zwhere_aux = jax.vmap(self._predict_zwhere)(q_li, prev_zwhere)  # (N, 3 or 5)
                    _, zpres_logit_aux = jax.vmap(
                        lambda q_, kk: self.pres_head(
                            q_, kk, tau=cfg.pres_tau, straight_through=False,
                            image_feats=feat_grid if cfg.pres_image_attn else None,
                        ),
                        in_axes=(0, 0),
                    )(q_li, k_dummy_per_slot)
                    aux_zwhere_layers.append(zwhere_aux)
                    aux_zpres_logit_layers.append(zpres_logit_aux)

            # Heads + decoder per slot.
            k_p = jax.random.fold_in(k, 1)
            k_w = jax.random.fold_in(k, 2)
            keys_p = jax.random.split(k_p, cfg.n_max)
            keys_w = jax.random.split(k_w, cfg.n_max)

            # Always produce the PREDICTED z_where so L_where can train it.
            zwhere_pred = jax.vmap(self._predict_zwhere)(q, prev_zwhere)              # (N, 3)
            render_zwhere = (
                render_zwhere_teacher if render_zwhere_teacher is not None else zwhere_pred
            )

            # We need a render_zpres to gate the appearance composite. If a teacher z_pres for
            # the CURRENT frame is provided, use it (padding slots get gated out). Otherwise we
            # use the model's predicted z_pres (computed inside _per_slot_head; until then we
            # use a placeholder of ones — _per_slot_head overwrites this when teacher is None).
            (zwhere, zpres, zpres_logit, zwhat, mu_w, lv_w, glimpse_feat,
             appear_canvas, mask_appear_canvas, mask_seg_canvas, g_post,
             mask_logit_patch, appear_patch) = jax.vmap(
                self._per_slot_head, in_axes=(0, 0, 0, 0, 0, 0, None, None)
            )(
                q,
                keys_p,
                keys_w,
                zwhere_pred,
                render_zwhere,
                render_zpres_teacher if render_zpres_teacher is not None else jnp.ones(cfg.n_max),
                image,
                feat_grid,
            )
            # When no teacher, swap in the predicted z_pres for the gate (we passed ones above
            # so _per_slot_head's mask_appear_canvas used 1.0; recompute with the real prediction).
            if render_zpres_teacher is None:
                # Re-multiply the mask_appear_canvas by the predicted z_pres ratio: since the
                # only place we used render_zpres was `sigmoid(mask_logit) * render_zpres` before
                # stn_write, and we wrote with render_zpres = 1, multiplying the canvas by
                # predicted zpres (possibly stop_gradient'd) reintroduces the gate.
                gate = (
                    jax.lax.stop_gradient(zpres) if cfg.stop_grad_recon_path else zpres
                )                                                                     # (N,)
                mask_appear_canvas = mask_appear_canvas * gate[:, None, None]

            # Composite: foreground (slot mixture) OVER background.
            num = jnp.sum(appear_canvas * mask_appear_canvas[..., None], axis=0)
            den = jnp.sum(mask_appear_canvas, axis=0)[..., None] + 1e-6
            fg = num / den                                                          # (H, W, 1)
            alpha_fg = jnp.clip(jnp.sum(mask_appear_canvas, axis=0), 0.0, 1.0)[..., None]
            if cfg.use_background:
                bg = bg_frame                                                       # (H, W, 1), in [0,1]
                composite = jnp.clip(alpha_fg * fg + (1.0 - alpha_fg) * bg, 0.0, 1.0)
            else:
                composite = jnp.clip(fg, 0.0, 1.0)

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
                g_post=g_post,
                mask_logit_patch=mask_logit_patch,
                appear_patch=appear_patch,
            )
            if aux_zwhere_layers:
                # Stack along a new layer axis → (L_aux, N, ...)
                out["z_where_aux"] = jnp.stack(aux_zwhere_layers, axis=0)
                out["z_pres_logit_aux"] = jnp.stack(aux_zpres_logit_layers, axis=0)
            return new_carry, out

        def step_structured(carry_s, inputs, refiner_module):
            """One-frame step for Neural EM / ISA — bypasses slot_transformer + heads.

            `refiner_module` is a callable (feat_grid, prev_z_where, prev_z_pres, prev_z_what)
            -> (z_where, z_pres, z_what) — i.e. NeuralEMStack or ISAStack (same signature).
            The per-slot decoder + STN + composite pipeline is unchanged. No teacher forcing.
            """
            prev_zwhere_ne, prev_zpres_ne, prev_zwhat_ne = carry_s
            feat_grid, image, _k_unused = inputs

            z_where, z_pres, z_what = refiner_module(
                feat_grid, prev_zwhere_ne, prev_zpres_ne, prev_zwhat_ne
            )

            (appear_canvas, mask_appear_canvas, mask_seg_canvas,
             mask_logit_patch, appear_patch, glimpse_feat) = jax.vmap(
                self._per_slot_decode_only, in_axes=(0, 0, 0, None)
            )(z_what, z_where, z_pres, image)

            num = jnp.sum(appear_canvas * mask_appear_canvas[..., None], axis=0)
            den = jnp.sum(mask_appear_canvas, axis=0)[..., None] + 1e-6
            fg = num / den
            alpha_fg = jnp.clip(jnp.sum(mask_appear_canvas, axis=0), 0.0, 1.0)[..., None]
            if cfg.use_background:
                composite = jnp.clip(alpha_fg * fg + (1.0 - alpha_fg) * bg_frame, 0.0, 1.0)
            else:
                composite = jnp.clip(fg, 0.0, 1.0)

            z_pres_logit = jax.scipy.special.logit(jnp.clip(z_pres, 1e-6, 1.0 - 1e-6))
            mu_w = z_what
            lv_w = jnp.full_like(z_what, -8.0)
            g_post = jnp.ones((cfg.n_max, 1))

            new_carry = (z_where, z_pres, z_what)
            out = dict(
                z_where=z_where, z_pres=z_pres, z_pres_logit=z_pres_logit,
                z_what=z_what, mu_w=mu_w, lv_w=lv_w,
                glimpse_feat=glimpse_feat,
                appear_pred=appear_canvas,
                mask_appear_pred=mask_appear_canvas,
                mask_seg_pred=mask_seg_canvas,
                composite=composite,
                g_post=g_post,
                mask_logit_patch=mask_logit_patch,
                appear_patch=appear_patch,
            )
            return new_carry, out

        # NOTE: We use a Python-unrolled loop rather than `jax.lax.scan` because the slot
        # transformer's parameters are created inside `setup()` and we need linen's parameter
        # tracking, which doesn't compose with `jax.lax.scan` without `nn.scan`. For T ≤ ~30
        # the unrolled graph is small enough not to matter.
        keys_t = jax.random.split(jax.random.fold_in(key, 1), T)
        outs = []

        if cfg.use_neural_em or cfg.use_isa:
            # z_pres init at 0.5 avoids log(0) deadlock in the E-step at t=0.
            prev_zpres0_s = jnp.full((cfg.n_max,), 0.5)
            if cfg.use_isa:
                refiner_mod = self.isa
                prev_zwhat0_s = self.z_what_init_isa
            else:
                refiner_mod = self.neural_em
                prev_zwhat0_s = self.z_what_init
            carry = (prev_zwhere0, prev_zpres0_s, prev_zwhat0_s)
            for t in range(T):
                carry, out_t = step_structured(
                    carry, (feats[t], video[t], keys_t[t]), refiner_mod
                )
                outs.append(out_t)
        else:
            carry = (slot_h0, prev_zwhere0, prev_zpres0, prev_zwhat0)
            for t in range(T):
                # Per-frame teacher slices (slot-index aligned to GT).
                if teacher_zwhere is not None:
                    prev_zwhere_anchor = teacher_zwhere[t - 1] if t > 0 else teacher_zwhere[0]
                    render_zwhere_teacher = teacher_zwhere[t]
                else:
                    prev_zwhere_anchor = None
                    render_zwhere_teacher = None
                if teacher_zpres is not None:
                    prev_zpres_gate = teacher_zpres[t - 1] if t > 0 else teacher_zpres[0]
                    render_zpres_teacher = teacher_zpres[t]
                else:
                    prev_zpres_gate = None
                    render_zpres_teacher = None

                carry, out_t = step(
                    carry,
                    (feats[t], video[t], keys_t[t]),
                    prev_zwhere_anchor,
                    prev_zpres_gate,
                    render_zwhere_teacher,
                    render_zpres_teacher,
                )
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
                g_post=traj["g_post"],                          # (T, N, K)
                mask_logit_patch=traj["mask_logit_patch"],      # (T, N, gh, gw, 1)
                appear_patch=traj["appear_patch"],              # (T, N, gh, gw, 1) — raw glimpse-space appearance
                **(
                    {
                        "z_where_aux": traj["z_where_aux"],         # (T, L_aux, N, ...)
                        "z_pres_logit_aux": traj["z_pres_logit_aux"],  # (T, L_aux, N)
                    }
                    if "z_where_aux" in traj
                    else {}
                ),
            ),
        )
