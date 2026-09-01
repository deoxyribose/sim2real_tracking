"""Autoregressive flagellum decoder.

Two heads on top of a shared U-Net encoder-decoder:
  1. Attachment head — per grid cell predicts (dy, dx, score) for the
     flagellum base position. Same idea as UNetEnergy but only the base.
  2. Knot generator — given (current_knot_position, last_tangent_angle),
     crops a rotated patch of the encoder feature map around the current
     knot, and predicts a categorical distribution over the NEXT step in
     polar form: (Δangle, step_len).

Inference: sample attachments, then autoregressively roll out K knots per
attachment. Each rollout is a plausible flagellum, sampled from the
categorical per-step distributions.

Why polar (Δangle, step_len) + rotated patches: the pattern "given a
canonical-orientation patch, where does the flagellum bend next" is a LOCAL
decision that transfers across sim and real. Whole-shape PCA prediction
requires the exact global shape to have appeared in the training basis,
which is the sim2real bottleneck we're trying to close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from sim2real.model.unet_energy import ConvBlock, Down, Up

Array = jnp.ndarray


@dataclass(frozen=True)
class UNetARConfig:
    T: int = 4
    H: int = 128
    W: int = 128
    grid_stride: int = 8              # attachment grid = H/8

    # Encoder-decoder
    base_channels: int = 48
    n_stages: int = 4
    use_bf16: bool = True

    # Feature map at input resolution for patch cropping.
    feature_channels: int = 32        # channels of the full-res feature map

    # Knot generator
    n_knots: int = 24                 # number of segments to emit after attachment
    patch_size: int = 24              # side-length of rotated feature-map crop
    n_angle_bins: int = 32            # Δangle bin count (over [-π/2, π/2])
    n_step_bins: int = 12             # step_len bin count
    angle_range: float = float(jnp.pi / 2)   # Δangle ∈ [-angle_range, angle_range]
    step_min: float = 0.5             # step_len bin edges span [step_min, step_max]
    step_max: float = 8.0             # up to 8-px steps (more room for wider flagella)

    # Attachment head
    n_attach_suggestions: int = 4     # attachments per grid cell

    # Noise input for stochasticity (kept for compatibility w/ energy_loss)
    noise_scale: float = 1.0
    noise_freq_cells: int = 4

    @property
    def grid_h(self) -> int: return self.H // self.grid_stride

    @property
    def grid_w(self) -> int: return self.W // self.grid_stride

    @property
    def angle_bin_edges(self) -> Array:
        return jnp.linspace(-self.angle_range, self.angle_range,
                             self.n_angle_bins + 1)

    @property
    def step_bin_edges(self) -> Array:
        return jnp.linspace(self.step_min, self.step_max, self.n_step_bins + 1)

    @property
    def angle_bin_centers(self) -> Array:
        e = self.angle_bin_edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def step_bin_centers(self) -> Array:
        e = self.step_bin_edges
        return 0.5 * (e[:-1] + e[1:])


# ---- Backbone --------------------------------------------------------------

class UNetARBackbone(nn.Module):
    """U-Net that outputs BOTH:
      - a full-resolution feature map (H, W, feature_channels) for patch crops
      - a grid-stride feature map for the attachment head
    """
    cfg: UNetARConfig

    @nn.compact
    def __call__(self, video: Array, noise: Array,
                 static_median: Array | None = None,
                 train: bool = True) -> tuple[Array, Array]:
        cfg = self.cfg
        dtype = jnp.bfloat16 if cfg.use_bf16 else jnp.float32
        if video.shape[1] == cfg.T:
            video = jnp.transpose(video, (0, 2, 3, 1))
        chans = [video, noise]
        if static_median is not None:
            chans.append(static_median)
        x = jnp.concatenate(chans, axis=-1).astype(dtype)

        skips = []
        for s in range(cfg.n_stages):
            skip, x = Down(cfg.base_channels * (2 ** s), dtype=dtype)(x, train)
            skips.append(skip)
        x = ConvBlock(cfg.base_channels * (2 ** cfg.n_stages), dtype=dtype)(x, train)

        # Full decoder up to input resolution
        for level in range(cfg.n_stages - 1, -1, -1):
            skip = skips[level]
            x = Up(cfg.base_channels * (2 ** level), dtype=dtype)(x, skip, train)

        full_res = nn.Conv(cfg.feature_channels, (1, 1),
                            dtype=dtype, param_dtype=jnp.float32)(x)
        full_res = full_res.astype(jnp.float32)

        # Grid features via average pooling from full-res
        gh = full_res.shape[1] // cfg.grid_stride
        gw = full_res.shape[2] // cfg.grid_stride
        grid = nn.avg_pool(full_res, (cfg.grid_stride, cfg.grid_stride),
                            strides=(cfg.grid_stride, cfg.grid_stride),
                            padding="SAME")
        return full_res, grid


class AttachmentHead(nn.Module):
    """1×1 conv on grid features → (n_sugg, 3) per grid cell."""
    cfg: UNetARConfig

    @nn.compact
    def __call__(self, grid_features: Array) -> Array:
        cfg = self.cfg
        n_out = cfg.n_attach_suggestions * 3   # (dy, dx, score)
        h = nn.Conv(64, (1, 1),
                     dtype=jnp.float32, param_dtype=jnp.float32)(grid_features)
        h = nn.gelu(h)
        h = nn.Conv(n_out, (1, 1),
                     dtype=jnp.float32, param_dtype=jnp.float32)(h)
        b, gh, gw, _ = h.shape
        return h.reshape(b, gh, gw, cfg.n_attach_suggestions, 3)


def unpack_attachment(pred: Array, offset_scale: float = 32.0) -> dict:
    """(..., 3) with (dy_raw, dx_raw, score_logit)."""
    return dict(
        dy=offset_scale * jnp.tanh(pred[..., 0]),
        dx=offset_scale * jnp.tanh(pred[..., 1]),
        score=pred[..., 2],
    )


# ---- Rotated patch crop ---------------------------------------------------

def _rotated_patch(feature_map: Array, center: Array, tangent: Array,
                   patch_size: int) -> Array:
    """Extract a `patch_size × patch_size` crop of feature_map, centered on
    `center=(y, x)`, rotated so the +x axis aligns with `tangent` direction.

    Args:
      feature_map: (H, W, C) single sample
      center: (2,) (y, x) in pixel coordinates
      tangent: scalar angle (radians). The patch is rotated so its +x axis
               points in `tangent` direction, i.e., the "in-front" direction
               of the flagellum end.
      patch_size: side length P
    Returns:
      (P, P, C) patch
    """
    H, W, C = feature_map.shape
    P = patch_size
    # Local coords: (i, j) in [-(P-1)/2, (P-1)/2].
    # We place the patch so that (0, +col) is the "forward" direction.
    ls = jnp.arange(P, dtype=jnp.float32) - (P - 1) / 2.0
    # ii is row (perpendicular to forward), jj is col (forward)
    ii, jj = jnp.meshgrid(ls, ls, indexing="ij")
    # Rotate: forward direction = tangent
    c, s = jnp.cos(tangent), jnp.sin(tangent)
    dy = ii * c + jj * s      # rows
    dx = -ii * s + jj * c     # cols
    ys = center[0] + dy
    xs = center[1] + dx
    # Bilinear sample per channel
    ys_c = jnp.clip(ys, 0.0, H - 1.0)
    xs_c = jnp.clip(xs, 0.0, W - 1.0)
    y0 = jnp.floor(ys_c).astype(jnp.int32)
    x0 = jnp.floor(xs_c).astype(jnp.int32)
    y1 = jnp.clip(y0 + 1, 0, H - 1)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    dy_f = ys_c - y0
    dx_f = xs_c - x0
    def gather(y, x):
        return feature_map[y, x]        # (P, P, C)
    v00 = gather(y0, x0); v01 = gather(y0, x1)
    v10 = gather(y1, x0); v11 = gather(y1, x1)
    wy = dy_f[..., None]; wx = dx_f[..., None]
    return ((1 - wy) * ((1 - wx) * v00 + wx * v01) +
            wy * ((1 - wx) * v10 + wx * v11))


def rotated_patch_batched(feature_map: Array, centers: Array, tangents: Array,
                           patch_size: int) -> Array:
    """Vmap over a batch of (center, tangent) pairs — feature_map is same for all."""
    return jax.vmap(_rotated_patch, in_axes=(None, 0, 0, None))(
        feature_map, centers, tangents, patch_size)


# ---- Knot generator -------------------------------------------------------

class KnotGenerator(nn.Module):
    """Predict categorical distribution over (Δangle, step_len) given a
    rotated feature-map patch centered on the current knot."""
    cfg: UNetARConfig

    @nn.compact
    def __call__(self, patch: Array) -> tuple[Array, Array]:
        cfg = self.cfg
        x = nn.Conv(64, (3, 3), padding="SAME",
                     dtype=jnp.float32, param_dtype=jnp.float32)(patch)
        x = nn.gelu(x)
        x = nn.Conv(64, (3, 3), padding="SAME",
                     dtype=jnp.float32, param_dtype=jnp.float32)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(128, dtype=jnp.float32,
                      param_dtype=jnp.float32)(x)
        x = nn.gelu(x)
        angle_logits = nn.Dense(cfg.n_angle_bins,
                                  dtype=jnp.float32,
                                  param_dtype=jnp.float32)(x)
        step_logits = nn.Dense(cfg.n_step_bins,
                                 dtype=jnp.float32,
                                 param_dtype=jnp.float32)(x)
        return angle_logits, step_logits


# ---- Noise sampler (reuse energy) ----------------------------------------

def sample_noise_map(key: jax.Array, cfg: UNetARConfig, temperature: float = 1.0) -> Array:
    small_h = max(1, cfg.H // (cfg.grid_stride * cfg.noise_freq_cells))
    small_w = max(1, cfg.W // (cfg.grid_stride * cfg.noise_freq_cells))
    small = jax.random.normal(key, (small_h, small_w, 1)) * (cfg.noise_scale * temperature)
    return jax.image.resize(small, (cfg.H, cfg.W, 1), method="linear")


def sample_batched_noise(key: jax.Array, batch_size: int, cfg: UNetARConfig,
                          temperature: float = 1.0) -> Array:
    keys = jax.random.split(key, batch_size)
    return jax.vmap(lambda k: sample_noise_map(k, cfg, temperature))(keys)


# ---- Convenience: encode GT skeleton into (Δangle, step_len) sequence -----

def encode_gt_polar_steps(skeleton: Array, cfg: UNetARConfig) -> tuple[Array, Array, Array, Array]:
    """Given a (K+1, 2) skeleton polyline (attachment + K knots), return:
        attachment:   (2,) initial (y, x)
        angles:       (K,) tangent angle *at the end of each step*
        d_angles:     (K,) target Δangle bin index per step (int32 in [0, n_bins))
        d_steps:      (K,) target step_len bin index per step (int32 in [0, n_bins))
    d_angles[0] uses tangent=0 as reference (initial "no prior direction").
    """
    K = skeleton.shape[0] - 1
    diffs = jnp.diff(skeleton, axis=0)                     # (K, 2) in (dy, dx)
    step_lens = jnp.linalg.norm(diffs, axis=-1)             # (K,)
    tangents = jnp.arctan2(diffs[:, 0], diffs[:, 1])        # (K,)
    prev_tangents = jnp.concatenate([jnp.zeros(1), tangents[:-1]])
    d_angle = jnp.arctan2(jnp.sin(tangents - prev_tangents),
                            jnp.cos(tangents - prev_tangents))    # wrap to [-π, π]

    # Bin into categoricals
    angle_edges = cfg.angle_bin_edges
    step_edges = cfg.step_bin_edges
    d_angle_clipped = jnp.clip(d_angle, angle_edges[0], angle_edges[-1] - 1e-6)
    step_clipped = jnp.clip(step_lens, step_edges[0], step_edges[-1] - 1e-6)
    d_angle_bin = jnp.searchsorted(angle_edges, d_angle_clipped, side="right") - 1
    step_bin = jnp.searchsorted(step_edges, step_clipped, side="right") - 1
    d_angle_bin = jnp.clip(d_angle_bin, 0, cfg.n_angle_bins - 1)
    step_bin = jnp.clip(step_bin, 0, cfg.n_step_bins - 1)

    return skeleton[0], tangents, d_angle_bin.astype(jnp.int32), step_bin.astype(jnp.int32)


def decode_polar_steps(attachment: Array, tangents: Array,
                       d_angle_idx: Array, d_step_idx: Array,
                       cfg: UNetARConfig) -> Array:
    """Reconstruct a (K+1, 2) skeleton from a sequence of (Δangle_bin, step_bin)."""
    d_angles = cfg.angle_bin_centers[d_angle_idx]
    steps = cfg.step_bin_centers[d_step_idx]
    # Compute tangents by cumulative sum (not needed at train time; but useful for eval)
    def scan_fn(carry, i):
        pos, tan = carry
        new_tan = tan + d_angles[i]
        step = steps[i]
        new_pos = jnp.stack([pos[0] + step * jnp.sin(new_tan),
                              pos[1] + step * jnp.cos(new_tan)])
        return (new_pos, new_tan), new_pos
    (_, _), positions = jax.lax.scan(
        scan_fn, (attachment, jnp.array(0.0)),
        jnp.arange(d_angle_idx.shape[0]))
    return jnp.concatenate([attachment[None], positions], axis=0)
