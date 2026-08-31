"""PCA basis for flagellum skeleton shapes.

We predict per-grid-cell:
    (dy, dx)      — attachment offset from cell center       (2 numbers)
    theta         — tangent angle at the base                (1 number)
    coeffs (16)   — PCA coefficients in a shape basis        (16 numbers)
    score         — confidence                               (1 number)
                                                        total 20 per cell

The PCA basis is fit on ``canonicalize_curve`` output — the curve translated
so its first point is at the origin and rotated so its base-tangent points
along +x. So the basis is pure SHAPE and doesn't need to span rotations.

This module provides:
  - canonicalize_curve / decanonicalize_curve
  - fit_pca (from a big pool of canonical curves)
  - a script that draws many sim clips and saves the fitted (mean, basis)
    to a .npz for the model to load.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip

Array = jnp.ndarray


# ---- Canonicalization -----------------------------------------------------

def canonicalize_curve(curve_yx: Array) -> tuple[Array, Array, Array]:
    """Translate so first point is at origin, rotate so base tangent → +x.

    Args:
      curve_yx: (K, 2) polyline in world (y, x).
    Returns:
      canon_yx: (K, 2) canonical curve (starts at (0,0) heading +x)
      base_yx:  (2,) — the original first point (attachment in world)
      theta:    scalar — the original base-tangent angle (from x-axis)
    """
    base = curve_yx[0]
    tangent = curve_yx[1] - curve_yx[0]
    theta = jnp.arctan2(tangent[0], tangent[1])
    # Rotate by -theta so tangent aligns with +x
    c, s = jnp.cos(-theta), jnp.sin(-theta)
    # Rotation for (y, x): y' = -x·sin(θ) + y·cos(θ);  x' = x·cos(θ) + y·sin(θ)
    # Here we rotate by angle -θ (so cos(-θ), sin(-θ)).
    shifted = curve_yx - base
    canon_y = shifted[:, 0] * c - shifted[:, 1] * s
    canon_x = shifted[:, 0] * s + shifted[:, 1] * c
    canon = jnp.stack([canon_y, canon_x], axis=-1)
    return canon, base, theta


def decanonicalize_curve(canon_yx: Array, base_yx: Array, theta: Array) -> Array:
    """Inverse of `canonicalize_curve`. Rotate by +theta and translate by base."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    world_y = canon_yx[:, 0] * c - canon_yx[:, 1] * s
    world_x = canon_yx[:, 0] * s + canon_yx[:, 1] * c
    return jnp.stack([world_y, world_x], axis=-1) + base_yx


# ---- PCA ------------------------------------------------------------------

def fit_pca(canonical_curves: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on a pool of canonical curves.

    Args:
      canonical_curves: (N, K, 2) array of canonical curves.
      n_components: number of PCA modes.
    Returns:
      mean:  (K, 2)  — mean canonical curve.
      basis: (n_components, K, 2)  — orthonormal PCA modes.
    """
    N, K, _ = canonical_curves.shape
    X = canonical_curves.reshape(N, K * 2)                     # (N, 2K)
    mean = X.mean(axis=0)                                       # (2K,)
    Xc = X - mean
    # SVD gives Xc = U · Σ · Vᵀ; the top rows of Vᵀ are principal directions.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis_flat = Vt[:n_components]                              # (nc, 2K)
    return mean.reshape(K, 2), basis_flat.reshape(n_components, K, 2)


def encode_pca(canon: Array, mean: Array, basis: Array) -> Array:
    """Project a canonical curve onto the PCA basis.
    Args:
      canon: (K, 2)          basis: (M, K, 2)          mean: (K, 2)
    Returns:
      (M,) PCA coefficients.
    """
    d = (canon - mean).reshape(-1)              # (2K,)
    B = basis.reshape(basis.shape[0], -1)       # (M, 2K)
    return B @ d


def decode_pca(coeffs: Array, mean: Array, basis: Array) -> Array:
    """Reconstruct a canonical curve from PCA coefficients.
    Args:
      coeffs: (M,)           basis: (M, K, 2)         mean: (K, 2)
    Returns:
      (K, 2) canonical curve.
    """
    return mean + jnp.einsum("m,mkd->kd", coeffs, basis)


# ---- Dataset gathering ---------------------------------------------------

def gather_canonical_curves(n_clips: int, cfg: DiverseSimConfig,
                            seed: int = 0, frames_per_clip: int = 4) -> np.ndarray:
    """Draw n_clips sim clips; from each, extract every alive flagellum at
    `frames_per_clip` evenly-spaced time indices; canonicalize; return pool.

    Returns:
      (M, K, 2) array of canonical curves.
    """
    key = jax.random.key(seed)

    @jax.jit
    def one(k):
        out = sample_clip(k, cfg)
        return dict(curves=out["curves"],
                    alive=out["flagella"]["alive"])

    pool = []
    for i in range(n_clips):
        key, sub = jax.random.split(key)
        out = one(sub)
        curves = np.asarray(out["curves"])         # (T, N, K, 2)
        alive = np.asarray(out["alive"])           # (N,)
        T, N, K, _ = curves.shape
        t_idxs = np.linspace(0, T - 1, frames_per_clip).astype(int)
        for t in t_idxs:
            for k in range(N):
                if not bool(alive[k]):
                    continue
                canon, _, _ = canonicalize_curve(jnp.asarray(curves[t, k]))
                pool.append(np.asarray(canon))
        if (i + 1) % max(1, n_clips // 20) == 0:
            print(f"  gathered {len(pool):>7d} curves from {i + 1}/{n_clips} clips",
                  flush=True)
    return np.stack(pool, axis=0)


# ---- CLI ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=10000,
                    help="sim clips to draw (each contributes ~4-8 curves)")
    ap.add_argument("--frames-per-clip", type=int, default=4)
    ap.add_argument("--n-components", type=int, default=16)
    ap.add_argument("--H", type=int, default=200,
                    help="sim canvas size (must match training input)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="data_cache/flagella_pca.npz")
    args = ap.parse_args()

    cfg = DiverseSimConfig(T=16, H=args.H, W=args.H)
    print(f"gathering canonical curves from {args.n_clips} sim clips...",
          flush=True)
    pool = gather_canonical_curves(args.n_clips, cfg, seed=args.seed,
                                    frames_per_clip=args.frames_per_clip)
    print(f"pool: {pool.shape}   arc-length p50 = "
          f"{np.median(np.linalg.norm(np.diff(pool, axis=1), axis=-1).sum(-1)):.2f} px",
          flush=True)

    print(f"fitting PCA (n_components={args.n_components})...", flush=True)
    mean, basis = fit_pca(pool, args.n_components)

    # Explained variance
    X = pool.reshape(pool.shape[0], -1) - mean.reshape(-1)
    B = basis.reshape(args.n_components, -1)
    total_var = float((X ** 2).sum(-1).mean())
    coeffs = X @ B.T
    per_mode_var = (coeffs ** 2).mean(0)
    cum_frac = per_mode_var.cumsum() / total_var
    print("PCA cumulative explained variance:")
    for i in range(args.n_components):
        print(f"  mode {i:2d}: cumfrac = {cum_frac[i]:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), mean=mean, basis=basis,
                          per_mode_var=per_mode_var, total_var=total_var,
                          config=dict(H=args.H, K=pool.shape[1],
                                       n_clips=args.n_clips,
                                       n_components=args.n_components))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
