"""Neural EM refiner: iterate structured latents (z_where, z_pres, z_what) directly.

Drop-in replacement for `SlotAttentionStack`. Each iteration is one E-step + M-step:

  E-step (soft pixel->slot responsibility):
    Q = MLP(pos_embed(z_where.pos) + z_what)       # slot query from structured state
    K = MLP(image_features + sinusoidal_pos_embed) # position-aware image keys
    logits[N, L] = (Q @ K.T) * scale + log(z_pres)[:, None]  # z_pres as prior
    resp = softmax(logits, axis=slots)             # each pixel claimed by one slot

  M-step:
    z_where.pos   = weighted centroid of pixel positions + small learned delta
    z_where.scale = weighted std of pixel positions + small learned delta
    z_what        = GRU( MLP(slot_feature) , prev_z_what )
    z_pres        = sigmoid( (mass - learned_threshold) / learned_temperature )

State carried across frames = (z_where, z_pres, z_what) directly. No opaque slot vector.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from sim2real.model.posenc import sinusoidal_2d

Array = jnp.ndarray

_S_CLIP = (0.02, 0.95)
_T_CLIP = 0.98


def _normalized_grid(h: int, w: int) -> Array:
    """Return (H*W, 2) pixel positions in [-1, 1] (y, x order matches sinusoidal_2d)."""
    ys = jnp.linspace(-1.0, 1.0, h)
    xs = jnp.linspace(-1.0, 1.0, w)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([xx, yy], axis=-1).reshape(h * w, 2)                                  # (L, 2)


def _slot_pos_embed(pos_xy: Array, d_pos: int) -> Array:
    """Sinusoidal position embedding for slot positions in [-1, 1]^2. Returns (N, d_pos)."""
    assert d_pos % 4 == 0, f"d_pos={d_pos} must be multiple of 4"
    dq = d_pos // 2
    div_term = jnp.exp(-jnp.log(10000.0) * jnp.arange(0, dq, 2) / dq)                      # (dq/2,)
    # pos_xy in [-1, 1]; rescale to arbitrary frequency space.
    x = pos_xy[:, 0:1] * 5.0                                                                # (N, 1)
    y = pos_xy[:, 1:2] * 5.0
    ex = jnp.concatenate([jnp.sin(x * div_term), jnp.cos(x * div_term)], axis=-1)          # (N, dq)
    ey = jnp.concatenate([jnp.sin(y * div_term), jnp.cos(y * div_term)], axis=-1)          # (N, dq)
    return jnp.concatenate([ex, ey], axis=-1)                                              # (N, d_pos)


def _pack_zwhere_from_centroid(centroid: Array, std: Array, theta_raw: Array) -> Array:
    """Convert (centroid_xy in [-1,1], std_xy in [0,1]) to 5-dim raw z_where.

    Uses inverse of the sigmoid/tanh transforms so the resulting raw form decodes back to
    the given centroid and scale.
    """
    tx_raw = jnp.arctanh(jnp.clip(centroid[:, 0], -_T_CLIP, _T_CLIP))                      # (N,)
    ty_raw = jnp.arctanh(jnp.clip(centroid[:, 1], -_T_CLIP, _T_CLIP))
    sx_raw = jax.scipy.special.logit(jnp.clip(std[:, 0], _S_CLIP[0], _S_CLIP[1]))
    sy_raw = jax.scipy.special.logit(jnp.clip(std[:, 1], _S_CLIP[0], _S_CLIP[1]))
    return jnp.stack([sx_raw, sy_raw, theta_raw, tx_raw, ty_raw], axis=-1)                 # (N, 5)


class NeuralEMRefiner(nn.Module):
    """One iteration of Neural EM on structured latents.

    Weights inside are shared across iterations when called in a loop.

    `use_bg_slot`: adds a phantom "null slot" to the softmax competition (with a learned
    scalar logit). Pixels not strongly wanted by real slots go to the null slot instead of
    being uniformly split. Prevents real slots from becoming background sinks (which
    contaminates their centroid). DETR uses the same idea via a "no object" class.
    """
    d_model: int
    d_pos: int = 32                          # position-embedding dimension
    z_where_delta_scale: float = 0.05        # correction magnitude in raw space (pos, scale)
    theta_delta_scale: float = 0.05          # separate delta scale for θ (per-iter reach)
    attn_temp: float = 1.0                   # softmax temperature; <1 = sharper responsibility
    use_bg_slot: bool = False                # add null slot for background pixels
    use_pca_theta: bool = False              # compute θ from PCA of responsibility covariance

    @nn.compact
    def __call__(self, K, V, pixel_pos, z_where, z_pres, z_what, prev_z_what,
                 bg_pixel_logits: Array | None = None,
                 resp_override: Array | None = None,
                 zwhere_override: Array | None = None,
                 zpres_override: Array | None = None):
        # K, V: (L, d_model); pixel_pos: (L, 2); z_where: (N, 5); z_pres: (N,); z_what: (N, Zw); prev_z_what: (N, Zw)
        Zw = z_what.shape[-1]

        # --- E-step ---
        # Query from structured state: pos_embed(z_where.pos) + z_what → d_model
        pos_xy = jnp.stack([jnp.tanh(z_where[:, 3]), jnp.tanh(z_where[:, 4])], axis=-1)    # (N, 2)
        pos_emb = _slot_pos_embed(pos_xy, self.d_pos)                                       # (N, d_pos)
        state_repr = jnp.concatenate([pos_emb, z_what], axis=-1)                           # (N, d_pos + Zw)
        state_repr = nn.LayerNorm(name="state_norm")(state_repr)
        Q = nn.Dense(self.d_model, use_bias=False, name="q_proj")(state_repr)              # (N, d_model)

        scale = self.d_model ** -0.5
        logits = (Q @ K.T) * scale / self.attn_temp + jnp.log(z_pres + 1e-6)[:, None]      # (N, L)
        # Optional: append a phantom "background" slot with learned scalar logit. Pixels
        # not strongly wanted by real slots go to bg instead of being uniformly split.
        if self.use_bg_slot:
            if bg_pixel_logits is not None:
                # Per-pixel bg logit, computed upstream from (feat, z_style). Overrides scalar.
                bg_row = bg_pixel_logits[None, :]                                          # (1, L)
            else:
                bg_logit = self.param("bg_logit", nn.initializers.constant(0.0), ())
                bg_row = jnp.broadcast_to(bg_logit, (1, logits.shape[-1]))                 # (1, L)
            logits_full = jnp.concatenate([logits, bg_row], axis=0)                        # (N+1, L)
            resp_full = jax.nn.softmax(logits_full, axis=0)                                # (N+1, L)
            resp = resp_full[:-1]                                                          # (N, L)
        else:
            # Softmax OVER SLOTS (each pixel is soft-claimed by one slot).
            resp = jax.nn.softmax(logits, axis=0)                                          # (N, L)

        # Optional oracle override: replace the computed responsibilities with a supplied
        # (N, L) map (e.g., downsampled GT masks). Downstream M-step then computes z_where,
        # z_pres, z_what from the forced responsibilities via the trained M-step weights.
        if resp_override is not None:
            resp = resp_override

        # --- M-step ---
        mass = resp.sum(axis=-1)                                                            # (N,)

        # Centroid + weighted std of pixel positions
        # centroid: (N, 2) = Σ_l resp[N,l] * pixel_pos[l] / mass[N]
        weighted_pos = resp @ pixel_pos                                                     # (N, 2)
        centroid = weighted_pos / (mass[:, None] + 1e-6)
        diff = pixel_pos[None] - centroid[:, None]                                          # (N, L, 2)
        weighted_var = (resp[:, :, None] * (diff ** 2)).sum(axis=1) / (mass[:, None] + 1e-6)  # (N, 2)
        std = jnp.sqrt(weighted_var + 1e-6)                                                 # (N, 2)

        # Aggregated slot features (for delta + z_what)
        slot_feat = (resp @ V) / (mass[:, None] + 1e-6)                                     # (N, d_model)
        slot_feat_norm = nn.LayerNorm(name="slot_feat_norm")(slot_feat)

        # z_where: centroid + small learned correction.
        x = nn.Dense(self.d_model, name="delta_hidden")(slot_feat_norm)
        x = nn.gelu(x)
        delta = nn.Dense(5, kernel_init=nn.initializers.zeros, name="delta_out")(x)         # (N, 5)
        if self.use_pca_theta:
            # 2x2 weighted covariance including off-diagonal.
            cxx = weighted_var[:, 0]                                                        # already computed above
            cyy = weighted_var[:, 1]
            cxy = (resp[:, :, None] * (diff[..., 0:1] * diff[..., 1:2])).sum(axis=1) \
                    .squeeze(-1) / (mass + 1e-6)                                            # (N,)
            # Principal-axis angle from covariance eigenvector. Range (-π/2, π/2).
            theta_pca = 0.5 * jnp.arctan2(2.0 * cxy, cxx - cyy + 1e-6)                      # (N,)
            # STN decodes θ = π * tanh(θ_raw), so invert: θ_raw = arctanh(θ / π).
            _PI_CLIP = 0.98
            theta_raw_pca = jnp.arctanh(jnp.clip(theta_pca / jnp.pi, -_PI_CLIP, _PI_CLIP))  # (N,)
            theta_raw = theta_raw_pca + self.theta_delta_scale * jnp.tanh(delta[:, 2])
        else:
            theta_raw = z_where[:, 2] + self.theta_delta_scale * jnp.tanh(delta[:, 2])
        base_zwhere = _pack_zwhere_from_centroid(centroid, std, theta_raw)                  # (N, 5)
        delta_xy = jnp.concatenate([delta[:, 0:2], jnp.zeros_like(delta[:, 2:3]), delta[:, 3:5]], axis=-1)
        z_where_new = base_zwhere + self.z_where_delta_scale * jnp.tanh(delta_xy)           # (N, 5)

        # z_what: GRU update. Cell state is the CURRENT-iteration z_what so iterations
        # actually accumulate refinement (matches Locatello Slot Attention). Previously we
        # passed `prev_z_what` (frame-boundary state), which reset the GRU each iteration and
        # meant only the last iteration's what_pre mattered — a real bug.
        what_pre = nn.Dense(Zw, name="what_pre")(slot_feat_norm)                            # (N, Zw)
        gru = nn.GRUCell(features=Zw, name="what_gru")
        z_what_new, _ = jax.vmap(gru)(z_what, what_pre)                                     # (N, Zw)
        # Residual MLP after GRU (Locatello 2020 has this — helps mixing).
        mlp_hidden = 4 * Zw
        z_what_new = z_what_new + nn.Dense(Zw, name="post_mlp_out")(
            nn.gelu(nn.Dense(mlp_hidden, name="post_mlp_in")(
                nn.LayerNorm(name="post_mlp_norm")(z_what_new)
            ))
        )

        # z_pres: sigmoid((mass - learned_thresh) / learned_temp)
        thresh = self.param("mass_thresh", nn.initializers.constant(1.0), ())               # scalar
        temp = self.param("mass_temp", nn.initializers.constant(1.0), ())
        z_pres_new = jax.nn.sigmoid((mass - thresh) / (jnp.abs(temp) + 0.1))                # (N,)

        # Optional oracle overrides: after M-step, replace z_where and/or z_pres with fed values.
        # z_what is left as computed by the trained M-step (from forced resp @ V, GRU, MLP).
        if zwhere_override is not None:
            z_where_new = zwhere_override
        if zpres_override is not None:
            z_pres_new = zpres_override

        return z_where_new, z_pres_new, z_what_new


class NeuralEMStack(nn.Module):
    """Full Neural EM stack: preproject image features, run N iterations, return latents.

    Parameters shared across iterations (single Refiner module reused).
    """
    n_max: int
    d_model: int
    z_what_dim: int
    n_iters: int = 3
    d_pos: int = 32
    attn_temp: float = 1.0
    use_bg_slot: bool = False
    use_pca_theta: bool = False
    theta_delta_scale: float = 0.05
    # When True (and `use_bg_slot`), the null-slot logit is computed per pixel from a small MLP
    # over (feats_l, z_style). Requires z_style to be passed in. Lets the bg slot distinguish
    # "obviously far-bg pixel" from "edge of a thin fg object" instead of a global scalar.
    bg_slot_per_pixel: bool = False

    @nn.compact
    def __call__(self, feat_grid, prev_z_where, prev_z_pres, prev_z_what,
                 z_style: Array | None = None,
                 resp_override: Array | None = None,
                 zwhere_override: Array | None = None,
                 zpres_override: Array | None = None):
        # feat_grid: (h, w, d). prev_* are the entering iteration state (from carry or init).
        h, w, d = feat_grid.shape
        L = h * w

        # Precompute K, V from image features (once per frame, shared across iters).
        pe = sinusoidal_2d(h, w, d)
        feats_flat = (feat_grid + pe).reshape(L, d)
        feats_flat = nn.LayerNorm(name="feat_norm")(feats_flat)
        K = nn.Dense(self.d_model, use_bias=False, name="k_proj")(feats_flat)               # (L, d_model)
        V = nn.Dense(self.d_model, use_bias=False, name="v_proj")(feats_flat)               # (L, d_model)

        pixel_pos = _normalized_grid(h, w)                                                  # (L, 2)

        # Per-pixel bg-slot logits (optional). Computed once per frame from (feats, z_style).
        bg_pixel_logits = None
        if self.use_bg_slot and self.bg_slot_per_pixel:
            assert z_style is not None, "bg_slot_per_pixel=True requires z_style"
            style_broadcast = jnp.broadcast_to(z_style[None], (L, z_style.shape[-1]))
            bg_in = jnp.concatenate([feats_flat, style_broadcast], axis=-1)                 # (L, d + Zs)
            bg_h = nn.gelu(nn.Dense(self.d_model // 2, name="bg_pix_h")(bg_in))
            bg_pixel_logits = nn.Dense(1, name="bg_pix_out",
                                       bias_init=nn.initializers.constant(0.0))(bg_h)[:, 0]  # (L,)

        refiner = NeuralEMRefiner(self.d_model, self.d_pos, attn_temp=self.attn_temp,
                                  use_bg_slot=self.use_bg_slot,
                                  use_pca_theta=self.use_pca_theta,
                                  theta_delta_scale=self.theta_delta_scale, name="refiner")

        z_where, z_pres, z_what = prev_z_where, prev_z_pres, prev_z_what
        traj_zwhere = []                    # per-iter outputs, length n_iters
        traj_zpres = []
        for _ in range(self.n_iters):
            z_where, z_pres, z_what = refiner(K, V, pixel_pos, z_where, z_pres, z_what, prev_z_what,
                                              bg_pixel_logits=bg_pixel_logits,
                                              resp_override=resp_override,
                                              zwhere_override=zwhere_override,
                                              zpres_override=zpres_override)
            traj_zwhere.append(z_where)
            traj_zpres.append(z_pres)
        # Stack trajectories along a new "layer" axis (matches DETR-style aux convention).
        zw_aux = jnp.stack(traj_zwhere[:-1], axis=0) if len(traj_zwhere) > 1 else None    # (n_iters-1, N, 5)
        zp_aux = jnp.stack(traj_zpres[:-1], axis=0) if len(traj_zpres) > 1 else None      # (n_iters-1, N)
        return z_where, z_pres, z_what, zw_aux, zp_aux
