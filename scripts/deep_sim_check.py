"""Single-sample deep diagnostic. For a chosen seed, show:
  raw | cell layer | pipette | flag layer | per-slot flag contributions | GT overlay
so we can attribute every dark feature to a specific source and catch bugs
where 'extra' flagella appear.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sim2real.sim.flagella_diverse import (
    DiverseSimConfig, sample_clip, sample_cells, sample_flagella, sample_pipette,
    render_cell_bodies, render_pipette, render_flagellum_frame,
    flagellum_curve_at_t, cells_interior_mask, add_bg_blobs, add_debris,
    slow_bg_texture, vignette, apply_psf, _temporal_box_blur,
)
from sim2real.sim.background import perlin_noise

OUT = Path("/home/frans/sim2real_tracking/runs/deep_sim_check.png")


def one_sample(seed: int, cfg: DiverseSimConfig):
    """Recreate `sample_clip` step-by-step and return every intermediate layer
    at the middle time step, plus per-slot flag contributions."""
    key = jax.random.key(1000 + seed)   # matches viz_sim_samples_grid
    k_scene, k_bg, k_dist, k_debris, k_noise, k_wave = jax.random.split(key, 6)
    k_cells, k_flag, k_pip, k_ctex, k_scale = jax.random.split(k_scene, 5)

    scene_scale = jax.random.uniform(k_scale, (), minval=cfg.scene_scale_min,
                                       maxval=cfg.scene_scale_max)
    cells = sample_cells(k_cells, cfg, scene_scale)
    flag = sample_flagella(k_flag, cfg, cells, scene_scale)
    pip = sample_pipette(k_pip, cfg, cells, scene_scale)

    k_c, k_f = jax.random.split(k_ctex)
    res = max(cfg.H, cfg.W)
    coarse = perlin_noise(k_c, res, cfg.cell_texture_steps_coarse)
    fine = perlin_noise(k_f, res, cfg.cell_texture_steps_fine)
    cell_texture = ((1.0 - cfg.cell_texture_fine_weight) * coarse
                     + cfg.cell_texture_fine_weight * fine)[:cfg.H, :cfg.W]

    T = cfg.T
    t_mid = T // 2
    t_norm_mid = t_mid / max(T - 1, 1)

    # Per-slot curve & contribution at mid
    def per_slot_curve(idx):
        params_i = {k: v[idx] if hasattr(v, "shape") and v.shape[:1] == (cfg.n_max_flagella,) else v
                    for k, v in flag.items()}
        return flagellum_curve_at_t(params_i, jnp.asarray(t_norm_mid), cfg)
    curves_mid = jax.vmap(per_slot_curve)(jnp.arange(cfg.n_max_flagella))  # (N, K, 2)

    per_slot = []
    for k in range(cfg.n_max_flagella):
        per = render_flagellum_frame(curves_mid[k], flag["width"][k],
                                       flag["amp"][k], flag["alive"][k], cfg)
        per_slot.append(np.asarray(per))

    cell_layer = np.asarray(render_cell_bodies(cells, cfg, jnp.asarray(t_norm_mid),
                                                cell_texture))
    pip_layer = np.asarray(render_pipette(pip, cfg, jnp.asarray(t_norm_mid)))
    cell_interior = np.asarray(cells_interior_mask(cells, cfg,
                                                     jnp.asarray(t_norm_mid)))
    flag_layer = sum(per_slot) * (1.0 - cell_interior)

    key = jax.random.key(1000 + seed)   # rebind for the second call
    out = sample_clip(key, cfg)   # for the final composite
    raw_mid = np.asarray(out["clip_raw"])[t_mid]
    med_mid = np.asarray(out["clip_median"])[t_mid]

    return dict(
        raw=raw_mid, med=med_mid,
        cell_layer=cell_layer, pip_layer=pip_layer, flag_layer=flag_layer,
        per_slot=per_slot,
        cells=cells, flag=flag, pip=pip,
        curves_mid=np.asarray(curves_mid),
    )


def rgb_signed(f, rng):
    return np.clip((f + rng) / (2 * rng), 0, 1)


def main():
    cfg = DiverseSimConfig(T=16, H=200, W=200)
    seeds = [0]     # zoom on the first sample the user's complaining about
    n_cols = 4 + cfg.n_max_flagella   # raw, med, cells, pipette, per-slot
    fig, axes = plt.subplots(len(seeds), n_cols,
                              figsize=(2 * n_cols + 1, 2.4 * len(seeds)),
                              squeeze=False)

    for r, s in enumerate(seeds):
        d = one_sample(s, cfg)
        alive = np.asarray(d["flag"]["alive"]).astype(int).tolist()
        amps = np.asarray(d["flag"]["amp"])

        rng_med = max(float(np.percentile(np.abs(d["med"]), 99.5)), 0.02)
        axes[r][0].imshow(d["raw"], cmap="gray")
        axes[r][0].set_title(f"[{s}] raw   alive={alive}", fontsize=8)
        axes[r][1].imshow(d["med"], cmap="seismic", vmin=-rng_med, vmax=rng_med)
        axes[r][1].set_title(f"− median   (rng={rng_med:.3f})", fontsize=8)

        rng_layer = max(float(np.abs(d["cell_layer"]).max()), 0.02)
        axes[r][2].imshow(d["cell_layer"], cmap="seismic",
                          vmin=-rng_layer, vmax=rng_layer)
        axes[r][2].set_title("cell layer alone", fontsize=8)

        rng_layer = max(float(np.abs(d["pip_layer"]).max()), 0.02)
        axes[r][3].imshow(d["pip_layer"], cmap="seismic",
                          vmin=-rng_layer, vmax=rng_layer)
        axes[r][3].set_title("pipette layer alone", fontsize=8)

        for k in range(cfg.n_max_flagella):
            per = d["per_slot"][k]
            slot_rng = max(abs(amps[k]) * 0.5, 0.01)
            axes[r][4 + k].imshow(per, cmap="seismic",
                                    vmin=-slot_rng, vmax=slot_rng)
            axes[r][4 + k].set_title(f"slot {k}  alive={alive[k]}  amp={amps[k]:.2f}",
                                       fontsize=7.5)

        for c in range(n_cols):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])

    fig.suptitle("Deep sim diagnostic — attribute every dark feature to its layer",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
