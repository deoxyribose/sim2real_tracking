"""Invariant Slot Attention (Biza et al., arXiv 2302.04973).

Each slot carries an explicit pose (pos, scale, theta) that acts as an equivariant reference
frame. Position encoding of pixel keys is transformed into each slot's frame BEFORE the
dot-product, so a translated / rotated / scaled object is picked up by the SAME slot regardless
of its absolute image position.

Structured latents (compatible with the rest of the pipeline):
  z_where ∈ R^5 = (sx_raw, sy_raw, theta_raw, tx_raw, ty_raw)   -- unconstrained
  z_what  ∈ R^Zw                                                -- pose-invariant appearance
  z_pres  ∈ [0, 1]                                              -- sigmoid((mass - thresh) / temp)

One iteration = one E-step + one M-step:

  E-step (equivariant attention, softmax OVER SLOTS):
    For slot i:  rel_i(l) = R(-theta_i) . (pixel_pos(l) - pos_i) / scale_i
                 K_i(l)   = K_base(l) + Wrel . sinusoidal(rel_i(l))
                 Q_i      = Wq . z_what_i                          # pose-invariant
                 logits_i(l) = Q_i . K_i(l) * d^-0.5 + log(z_pres_i + eps)
                 resp = softmax_over_slots(logits)                 # (N, L)

  M-step:
    z_where.pos    = weighted centroid of pixel positions
    z_where.scale  = weighted std along canonical axes
    z_where.theta  = principal-component angle: 0.5 * atan2(2*Cxy, Cxx - Cyy)
    z_what         = GRU(z_what, W_what . (resp @ V))
    z_pres         = sigmoid((mass - learned_thresh) / learned_temp)

Notes:
- The pose factorization mirrors our existing z_where convention exactly, so downstream
  glimpse decoder + STN write can consume ISA outputs without any translation shim.
- No absolute-position PE is added to the base keys — all positional information enters
  through the per-slot equivariant PE. This is the load-bearing part of the paper.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from sim2real.model.posenc import sinusoidal_2d

Array = jnp.ndarray


_S_CLIP = (0.02, 0.95)     # sigmoid range for scale (matches neural_em)
_T_CLIP = 0.98             # tanh range for translation (matches neural_em)
_REL_FREQ = 5.0            # frequency-space scale for the relative PE


def _normalized_grid(h: int, w: int) -> Array:
    ys = jnp.linspace(-1.0, 1.0, h)
    xs = jnp.linspace(-1.0, 1.0, w)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([xx, yy], axis=-1).reshape(h * w, 2)                                  # (L, 2)


def _fourier_from_coords(coords: Array, dim: int) -> Array:
    """Sinusoidal encoding for arbitrary (x, y) coords in R^2. Shape (..., 2) -> (..., dim)."""
    assert dim % 4 == 0, f"dim={dim} must be multiple of 4"
    dq = dim // 2
    div_term = jnp.exp(-jnp.log(10000.0) * jnp.arange(0, dq, 2) / dq)                      # (dq/2,)
    x = coords[..., 0:1] * _REL_FREQ
    y = coords[..., 1:2] * _REL_FREQ
    ex = jnp.concatenate([jnp.sin(x * div_term), jnp.cos(x * div_term)], axis=-1)          # (..., dq)
    ey = jnp.concatenate([jnp.sin(y * div_term), jnp.cos(y * div_term)], axis=-1)          # (..., dq)
    return jnp.concatenate([ex, ey], axis=-1)                                              # (..., dim)


def _pack_zwhere(centroid: Array, std: Array, theta_raw: Array) -> Array:
    """(centroid_xy in [-1,1], std_xy in [0,1], theta_raw) -> 5-dim raw z_where."""
    tx_raw = jnp.arctanh(jnp.clip(centroid[:, 0], -_T_CLIP, _T_CLIP))
    ty_raw = jnp.arctanh(jnp.clip(centroid[:, 1], -_T_CLIP, _T_CLIP))
    sx_raw = jax.scipy.special.logit(jnp.clip(std[:, 0], _S_CLIP[0], _S_CLIP[1]))
    sy_raw = jax.scipy.special.logit(jnp.clip(std[:, 1], _S_CLIP[0], _S_CLIP[1]))
    return jnp.stack([sx_raw, sy_raw, theta_raw, tx_raw, ty_raw], axis=-1)                 # (N, 5)


class ISARefiner(nn.Module):
    """One iteration of Invariant Slot Attention on structured latents."""

    d_model: int
    d_pos: int = 32
    z_where_delta_scale: float = 0.05
    scale_min: float = 0.05
    scale_max: float = 1.0

    @nn.compact
    def __call__(self, K_base, V, pixel_pos, z_where, z_pres, z_what, prev_z_what):
        # K_base, V: (L, d_model); pixel_pos: (L, 2); z_where: (N, 5); z_pres: (N,);
        # z_what, prev_z_what: (N, Zw)
        Zw = z_what.shape[-1]

        # --- Decode pose from z_where (unconstrained -> constrained) ---
        sx = jax.nn.sigmoid(z_where[:, 0])                                                 # (N,) in (0,1)
        sy = jax.nn.sigmoid(z_where[:, 1])
        theta = z_where[:, 2]                                                              # (N,) unbounded
        tx = jnp.tanh(z_where[:, 3])
        ty = jnp.tanh(z_where[:, 4])
        pos = jnp.stack([tx, ty], axis=-1)                                                 # (N, 2)

        # --- Equivariant PE: per-slot rotated & scaled pixel offset ---
        # d[i, l, :] = pixel_pos[l] - pos[i]                                (N, L, 2)
        d = pixel_pos[None] - pos[:, None]                                                 # (N, L, 2)
        # Rotation by -theta_i: [[cos, sin], [-sin, cos]] @ (dx, dy)
        cos_t = jnp.cos(-theta)[:, None]                                                   # (N, 1)
        sin_t = jnp.sin(-theta)[:, None]                                                   # (N, 1)
        rot_x = cos_t * d[..., 0] - sin_t * d[..., 1]                                      # (N, L)
        rot_y = sin_t * d[..., 0] + cos_t * d[..., 1]                                      # (N, L)
        # Normalize by (clipped) per-axis scale — small slots see local detail, big slots see far context.
        sx_c = jnp.clip(sx, self.scale_min, self.scale_max)[:, None]                       # (N, 1)
        sy_c = jnp.clip(sy, self.scale_min, self.scale_max)[:, None]
        rel_x = rot_x / sx_c
        rel_y = rot_y / sy_c
        rel = jnp.stack([rel_x, rel_y], axis=-1)                                           # (N, L, 2)

        rel_pe = _fourier_from_coords(rel, self.d_pos)                                     # (N, L, d_pos)
        rel_pe = nn.LayerNorm(name="rel_pe_norm")(rel_pe)
        rel_proj = nn.Dense(self.d_model, use_bias=False, name="rel_proj")(rel_pe)         # (N, L, d_model)
        K_slot = K_base[None] + rel_proj                                                   # (N, L, d_model)

        # --- Query from z_what (pose-invariant) ---
        z_what_norm = nn.LayerNorm(name="what_norm")(z_what)
        Q = nn.Dense(self.d_model, use_bias=False, name="q_proj")(z_what_norm)             # (N, d_model)

        # --- Attention: softmax over slots, z_pres as prior ---
        scale = self.d_model ** -0.5
        logits = jnp.einsum("nd,nld->nl", Q, K_slot) * scale
        logits = logits + jnp.log(z_pres + 1e-6)[:, None]                                  # (N, L)
        resp = jax.nn.softmax(logits, axis=0)                                              # (N, L)

        # --- M-step ---
        mass = resp.sum(axis=-1)                                                           # (N,)
        weighted_pos = resp @ pixel_pos                                                    # (N, 2)
        centroid = weighted_pos / (mass[:, None] + 1e-6)                                   # (N, 2)

        # Weighted 2x2 covariance for scale + orientation
        diff2 = pixel_pos[None] - centroid[:, None]                                        # (N, L, 2)
        w = resp                                                                            # (N, L)
        cxx = (w * diff2[..., 0] * diff2[..., 0]).sum(axis=-1) / (mass + 1e-6)             # (N,)
        cyy = (w * diff2[..., 1] * diff2[..., 1]).sum(axis=-1) / (mass + 1e-6)
        cxy = (w * diff2[..., 0] * diff2[..., 1]).sum(axis=-1) / (mass + 1e-6)
        # Principal-component angle. For a 2x2 real-symmetric matrix the eigenvector angle is
        # 0.5 * atan2(2 * cxy, cxx - cyy). Falls back to 0 gracefully for isotropic clusters.
        theta_new_raw = 0.5 * jnp.arctan2(2.0 * cxy, cxx - cyy + 1e-6)                     # (N,)
        # Axis-aligned per-axis std (ignore off-diagonal for scale — orientation absorbed by theta)
        std = jnp.sqrt(jnp.stack([cxx, cyy], axis=-1) + 1e-6)                              # (N, 2)

        # Aggregated slot feature for the small residual delta + z_what update
        slot_feat = (resp @ V) / (mass[:, None] + 1e-6)                                    # (N, d_model)
        slot_feat_norm = nn.LayerNorm(name="slot_feat_norm")(slot_feat)

        # z_where = moments (from resp) + small learned zero-init delta (all 5 channels)
        base_zwhere = _pack_zwhere(centroid, std, theta_new_raw)                           # (N, 5)
        h_delta = nn.gelu(nn.Dense(self.d_model, name="delta_hidden")(slot_feat_norm))
        delta = nn.Dense(5, kernel_init=nn.initializers.zeros, name="delta_out")(h_delta)  # (N, 5)
        z_where_new = base_zwhere + self.z_where_delta_scale * jnp.tanh(delta)

        # z_what: GRU update on current-iteration state (iterations accumulate refinement,
        # matching Locatello Slot Attention). Followed by residual MLP.
        what_pre = nn.Dense(Zw, name="what_pre")(slot_feat_norm)                           # (N, Zw)
        gru = nn.GRUCell(features=Zw, name="what_gru")
        z_what_new, _ = jax.vmap(gru)(z_what, what_pre)                                    # (N, Zw)
        mlp_hidden = 4 * Zw
        z_what_new = z_what_new + nn.Dense(Zw, name="post_mlp_out")(
            nn.gelu(nn.Dense(mlp_hidden, name="post_mlp_in")(
                nn.LayerNorm(name="post_mlp_norm")(z_what_new)
            ))
        )

        # z_pres from attention mass (learned threshold + temperature)
        thresh = self.param("mass_thresh", nn.initializers.constant(1.0), ())
        temp = self.param("mass_temp", nn.initializers.constant(1.0), ())
        z_pres_new = jax.nn.sigmoid((mass - thresh) / (jnp.abs(temp) + 0.1))               # (N,)

        return z_where_new, z_pres_new, z_what_new


class ISAStack(nn.Module):
    """Full ISA stack: preproject image features, run N iterations, return latents.

    Weights inside the refiner are shared across iterations (single module reused).

    `abs_pe_weight`: if > 0, add sinusoidal absolute PE (of the image feature grid) to feats
    before the k_proj — breaks strict equivariance but gives the attention a fallback signal
    when the equivariant per-slot PE fails (bad slot pose init). The ISA paper's reference
    code does something similar with a small learned weight.
    """

    n_max: int
    d_model: int
    z_what_dim: int
    n_iters: int = 3
    d_pos: int = 32
    abs_pe_weight: float = 0.0

    @nn.compact
    def __call__(self, feat_grid, prev_z_where, prev_z_pres, prev_z_what):
        h, w, d = feat_grid.shape
        L = h * w

        # By default: NO absolute PE — ISA's whole point is that positional info flows through
        # the per-slot equivariant PE inside the refiner. With abs_pe_weight > 0, mix in a
        # small amount of absolute PE as a fallback signal (breaks strict equivariance).
        if self.abs_pe_weight > 0.0:
            pe_abs = sinusoidal_2d(h, w, d)                                                # (h, w, d)
            feat_grid = feat_grid + self.abs_pe_weight * pe_abs

        feats_flat = feat_grid.reshape(L, d)
        feats_flat = nn.LayerNorm(name="feat_norm")(feats_flat)
        K_base = nn.Dense(self.d_model, use_bias=False, name="k_proj")(feats_flat)         # (L, d_model)
        V = nn.Dense(self.d_model, use_bias=False, name="v_proj")(feats_flat)              # (L, d_model)

        pixel_pos = _normalized_grid(h, w)                                                 # (L, 2)
        refiner = ISARefiner(self.d_model, self.d_pos, name="refiner")

        z_where, z_pres, z_what = prev_z_where, prev_z_pres, prev_z_what
        for _ in range(self.n_iters):
            z_where, z_pres, z_what = refiner(
                K_base, V, pixel_pos, z_where, z_pres, z_what, prev_z_what
            )
        # No per-iter deep supervision for ISA yet; return None for aux slots to match
        # NeuralEMStack signature.
        return z_where, z_pres, z_what, None, None
