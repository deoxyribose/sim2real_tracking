"""Smoke test the DIR pipeline: generate a sim clip, run DIR on it, print
solution stats. If this survives, we know all pieces plumb together."""
from __future__ import annotations

import jax
import numpy as np

from sim2real.dir.run_dir import DIRRunConfig, run_dir_on_clips
from sim2real.dir.build_problem import BuildConfig
from sim2real.dir.solve import SolveConfig
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def main():
    # Draw a sim clip — 32 frames at 256×256 to match the model
    cfg = DiverseSimConfig(T=32, H=256, W=256)
    out = sample_clip(jax.random.key(1234), cfg)
    residual = np.asarray(out["clip_median"])           # (T, H, W)
    print(f"residual clip {residual.shape}   n_alive_flag={int(out['flagella']['alive'].sum())}",
          flush=True)

    dir_cfg = DIRRunConfig(
        n_noise_draws=6,
        score_thresh=0.05,
        top_k_per_draw=24,
        build=BuildConfig(
            cost_mode="score_only",
            cost_scale=100.0,
            pick_cost_base=3.0,     # only preds with score > pick_cost_base / score_bonus picked
            score_bonus=30.0,       # → cutoff at score ≈ 0.10
            birth_cost=0.5,         # cheap track ends
            death_cost=0.5,
            link_max_dist=25.0,
            link_cost_scale=0.2,    # per-px cost of a link
        ),
        solve=SolveConfig(time_limit_s=30.0),
    )
    result = run_dir_on_clips(
        ckpt_path="runs/energy_v2/ckpt_step010000.pkl",
        pca_path="data_cache/flagella_pca.npz",
        clips=residual,
        dir_cfg=dir_cfg,
    )
    print("selected:", result["solution"]["selected_indices"][:20])
    print("tracks:", [len(t) for t in result["solution"]["tracks"]])


if __name__ == "__main__":
    main()
