"""Fresh U-Net + grid output head for energy-score training.

Input encoding
--------------
Video (T, H, W) + a 1D noise channel z (H, W) concatenated along channel
axis → (H, W, T + 1). The noise map is drawn per-clip at inference to make
the network stochastic; the energy-score loss trains a calibrated posterior.

Output head (per grid cell, stride 8)
-------------------------------------
`n_suggestions` predictions per cell, each of shape (20,) —
   dy, dx        cell-center-relative attachment offset (in canonical px)
   theta         base-tangent angle
   pca[16]       shape coefficients in the fitted PCA basis
   score         confidence (raw logit)

Reconstruct a world-frame skeleton from a prediction as:
   attach_world = cell_center + (dy, dx)
   canon_curve  = mean + basis @ pca
   world_curve  = decanonicalize_curve(canon_curve, attach_world, theta)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jnp.ndarray


# ---- Config ---------------------------------------------------------------

@dataclass(frozen=True)
class UNetConfig:
    T: int = 16                          # input frames
    H: int = 256
    W: int = 256
    grid_stride: int = 8                 # 256/8 = 32 grid cells per side
    n_suggestions: int = 4               # predictions per grid cell
    n_pca: int = 16                      # PCA coeffs per prediction
    base_channels: int = 32
    n_stages: int = 4                    # downsampling stages
    # noise channel: low-freq perlin-like scalar map per clip
    noise_scale: float = 1.0
    noise_freq_cells: int = 4            # coarser noise = less overfitting
    use_bf16: bool = True                # mixed-precision (bf16 compute, fp32 params)

    @property
    def n_out_per_pred(self) -> int:
        # (dy, dx, theta, pca[n_pca], width_raw, amp_raw, score)
        # width_raw → width_px via softplus; amp_raw → signed amp via tanh.
        return 3 + self.n_pca + 3

    @property
    def grid_h(self) -> int: return self.H // self.grid_stride

    @property
    def grid_w(self) -> int: return self.W // self.grid_stride


# ---- U-Net building blocks -----------------------------------------------

class ConvBlock(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool):
        # dtype = compute dtype (bf16 possible); param_dtype = master (fp32).
        x = nn.Conv(self.features, (3, 3), padding="SAME", use_bias=False,
                    dtype=self.dtype, param_dtype=jnp.float32)(x)
        # GroupNorm runs stats in fp32 for stability but casts output to `dtype`.
        x = nn.GroupNorm(num_groups=min(8, self.features),
                          dtype=self.dtype, param_dtype=jnp.float32)(x)
        x = nn.gelu(x)
        x = nn.Conv(self.features, (3, 3), padding="SAME", use_bias=False,
                    dtype=self.dtype, param_dtype=jnp.float32)(x)
        x = nn.GroupNorm(num_groups=min(8, self.features),
                          dtype=self.dtype, param_dtype=jnp.float32)(x)
        x = nn.gelu(x)
        return x


class Down(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train):
        skip = ConvBlock(self.features, dtype=self.dtype)(x, train)
        pooled = nn.avg_pool(skip, (2, 2), strides=(2, 2), padding="SAME")
        return skip, pooled


class Up(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, skip, train):
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        x = jnp.concatenate([x, skip], axis=-1)
        x = ConvBlock(self.features, dtype=self.dtype)(x, train)
        return x


# ---- Main model -----------------------------------------------------------

class UNetEnergy(nn.Module):
    cfg: UNetConfig

    @nn.compact
    def __call__(self, video: Array, noise: Array, train: bool = True) -> Array:
        """Forward pass.

        Args:
          video: (B, T, H, W) or (B, H, W, T) — grayscale intensity in [0, 1].
                 Auto-transposed if T comes first.
          noise: (B, H, W, 1) — per-clip noise map (temperature-scaled at inference).
          train: dropout/BN flag (currently unused — GroupNorm has no state).

        Returns:
          preds: (B, grid_h, grid_w, n_suggestions, n_out_per_pred)
        """
        cfg = self.cfg
        compute_dtype = jnp.bfloat16 if cfg.use_bf16 else jnp.float32
        # Normalize video shape → (B, H, W, T)
        if video.shape[1] == cfg.T:
            video = jnp.transpose(video, (0, 2, 3, 1))    # (B, T, H, W) → (B, H, W, T)
        x = jnp.concatenate([video, noise], axis=-1)      # (B, H, W, T + 1)
        x = x.astype(compute_dtype)

        # Encoder
        skips = []
        for s in range(cfg.n_stages):
            skip, x = Down(cfg.base_channels * (2 ** s), dtype=compute_dtype)(x, train)
            skips.append(skip)

        # Bottleneck
        x = ConvBlock(cfg.base_channels * (2 ** cfg.n_stages),
                       dtype=compute_dtype)(x, train)

        # Decoder — go up until we hit `grid_stride`
        # After `n_stages` downsamples we're at stride 2^n_stages relative to input.
        # We stop upsampling when stride == cfg.grid_stride.
        allowed = {2 ** s for s in range(cfg.n_stages + 1)}
        assert cfg.grid_stride in allowed, \
            f"grid_stride={cfg.grid_stride} must be a power of 2 within n_stages"
        # Pre-compute number of ups at Python level (no traced values)
        n_up = 0
        stride = 2 ** cfg.n_stages
        while stride > cfg.grid_stride:
            n_up += 1
            stride //= 2
        for i in range(n_up):
            level = cfg.n_stages - 1 - i
            skip = skips[level]
            x = Up(cfg.base_channels * (2 ** level), dtype=compute_dtype)(x, skip, train)

        # Grid head: cast back to fp32 before the final projection so the
        # loss computation stays in fp32 (Chamfer distances + softmax + BCE).
        x = x.astype(jnp.float32)
        out_channels = cfg.n_suggestions * cfg.n_out_per_pred
        x = nn.Conv(out_channels, (1, 1),
                     dtype=jnp.float32, param_dtype=jnp.float32)(x)
        b, gh, gw, _ = x.shape
        return x.reshape(b, gh, gw, cfg.n_suggestions, cfg.n_out_per_pred)


# ---- Prediction decoding --------------------------------------------------

def unpack_pred(pred: Array,
                offset_scale: float = 32.0,
                pca_scale: float = 3.0,
                width_scale: float = 4.0,
                amp_scale: float = 0.5) -> dict:
    """Split per-prediction vector into named fields with bounded ranges so
    runaway logits don't explode the curve reconstruction.

    Layout in `pred[..., :]`: (dy, dx, theta, pca[n_pca], width_raw, amp_raw, score)

    Args:
      offset_scale: |dy|, |dx| bounded to this many pixels from cell center.
      pca_scale:    PCA coefficients bounded to ±pca_scale (σ-normalized basis).
      width_scale:  soft-plus max magnitude on width (px). Real widths ≤4 px.
      amp_scale:    tanh(x)*amp_scale bounds signed amplitude in [0,1] units.
    """
    n_pca_start = 3
    n_pca_end = pred.shape[-1] - 3    # last three are width, amp, score
    return dict(
        dy=offset_scale * jnp.tanh(pred[..., 0]),
        dx=offset_scale * jnp.tanh(pred[..., 1]),
        theta=jnp.pi * jnp.tanh(pred[..., 2]),
        pca=pca_scale * jnp.tanh(pred[..., n_pca_start:n_pca_end]),
        # positive-only via softplus, scaled + clipped
        width=width_scale * jax.nn.sigmoid(pred[..., -3]),
        # signed via tanh, in [0,1] intensity units
        amp=amp_scale * jnp.tanh(pred[..., -2]),
        score=pred[..., -1],       # sigmoid applied externally
    )


def grid_cell_centers(cfg: UNetConfig) -> Array:
    """(grid_h, grid_w, 2) — pixel-space (y, x) center of each grid cell."""
    ys = jnp.arange(cfg.grid_h) * cfg.grid_stride + cfg.grid_stride / 2.0
    xs = jnp.arange(cfg.grid_w) * cfg.grid_stride + cfg.grid_stride / 2.0
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([yy, xx], axis=-1)


def decode_curves(pred: Array, cfg: UNetConfig,
                  pca_mean: Array, pca_basis: Array) -> Array:
    """Turn raw predictions into world-space skeleton curves.

    Args:
      pred: (B, grid_h, grid_w, n_suggestions, n_out_per_pred).
      pca_mean: (K, 2); pca_basis: (n_pca, K, 2).
    Returns:
      curves: (B, grid_h, grid_w, n_suggestions, K, 2) world-space skeletons.
    """
    f = unpack_pred(pred)
    centers = grid_cell_centers(cfg)                    # (gh, gw, 2)
    attach = centers[None, :, :, None, :] + jnp.stack([f["dy"], f["dx"]], -1)
    # Canonical curve = mean + coeffs · basis
    canon = pca_mean + jnp.einsum("...m,mkd->...kd", f["pca"], pca_basis)
    # Rotate by theta, translate by attach
    c, s = jnp.cos(f["theta"]), jnp.sin(f["theta"])
    canon_y = canon[..., 0]
    canon_x = canon[..., 1]
    world_y = canon_y * c[..., None] - canon_x * s[..., None] + attach[..., 0:1]
    world_x = canon_y * s[..., None] + canon_x * c[..., None] + attach[..., 1:2]
    return jnp.concatenate([world_y[..., None], world_x[..., None]], axis=-1)


# ---- Noise-map sampler ---------------------------------------------------

def sample_noise_map(key: jax.Array, cfg: UNetConfig, temperature: float = 1.0) -> Array:
    """Draw a per-clip low-frequency noise map z ∈ (H, W, 1).

    We upsample a small random Gaussian tensor to full resolution — cheap
    Perlin-like low-freq noise. `temperature` scales the amplitude (>1 at
    inference for over-diverse sampling)."""
    small_h = max(1, cfg.H // (cfg.grid_stride * cfg.noise_freq_cells))
    small_w = max(1, cfg.W // (cfg.grid_stride * cfg.noise_freq_cells))
    small = jax.random.normal(key, (small_h, small_w, 1)) * (cfg.noise_scale * temperature)
    z = jax.image.resize(small, (cfg.H, cfg.W, 1), method="linear")
    return z


def sample_batched_noise(key: jax.Array, batch_size: int, cfg: UNetConfig,
                          temperature: float = 1.0) -> Array:
    """(B, H, W, 1) — per-example noise maps."""
    keys = jax.random.split(key, batch_size)
    return jax.vmap(lambda k: sample_noise_map(k, cfg, temperature))(keys)
