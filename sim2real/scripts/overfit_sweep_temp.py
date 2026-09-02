"""Sweep temperature at eval on a trained ckpt to check the exposure-bias gap."""
import argparse, pickle
from pathlib import Path
import jax, numpy as np

from sim2real.eval_v2.coverage import _chamfer_polylines
from sim2real.model.unet_ar import (
    AttachmentHead, KnotGenerator, UNetARBackbone, UNetARConfig,
)
from sim2real.scripts.ar_batched import make_sampler, sample_pool_one_clip
from sim2real.sim.flagella_diverse import DiverseSimConfig, sample_clip


def build_batch(key, sim_cfg, B):
    keys = jax.random.split(key, B)
    outs = jax.vmap(lambda k: sample_clip(k, sim_cfg))(keys)
    return outs


def load_ckpt(path):
    d = pickle.loads(Path(path).read_bytes())
    cfg = UNetARConfig(**{k: v for k, v in d["cfg"].items()
                            if k in UNetARConfig.__dataclass_fields__})
    return d["params"], cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--n-attach", type=int, default=16)
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--tta-angles", nargs="+", type=float,
                    default=[0.0, -10.0, 10.0])
    ap.add_argument("--temperatures", nargs="+", type=float,
                    default=[0.0, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--coverage", type=float, default=6.0)
    args = ap.parse_args()

    params, cfg = load_ckpt(args.ckpt)
    backbone = UNetARBackbone(cfg=cfg)
    attach_head = AttachmentHead(cfg=cfg)
    knot_gen = KnotGenerator(cfg=cfg)
    sim_cfg = DiverseSimConfig(T=cfg.T, H=cfg.H, W=cfg.W,
                                 sigma_scale_residual=False)  # v8 default

    # Rebuild the fixed batch
    bkey = jax.random.key(args.batch_seed)
    outs = build_batch(bkey, sim_cfg, args.batch_size)
    clip = np.asarray(outs["clip_median"])
    smed = np.asarray(outs["temporal_median"])
    curves = np.asarray(outs["curves"][:, cfg.T // 2])
    valid  = np.asarray(outs["flagella"]["alive"])
    gts_all = [[curves[b, i] for i in range(curves.shape[1])]
                 for b in range(args.batch_size)]

    for T in args.temperatures:
        sampler = make_sampler(cfg, backbone, attach_head, knot_gen, temperature=T)
        key = jax.random.key(999)
        n_hit, n_tot = 0, 0
        for b in range(args.batch_size):
            rollouts, key = sample_pool_one_clip(
                params, backbone, attach_head, knot_gen, cfg,
                clip[b], smed[b],
                list(args.tta_angles), flips=(False, True),
                n_draws=args.n_draws, n_attach=args.n_attach,
                n_rollouts=args.n_rollouts,
                score_thresh=0.02, key=key, _sampler=sampler)
            for i, g in enumerate(gts_all[b]):
                if not bool(valid[b, i]):
                    continue
                n_tot += 1
                if not rollouts: continue
                dists = [_chamfer_polylines(rl, g) for rl in rollouts]
                if min(dists) <= args.coverage: n_hit += 1
        n_rollouts_per_clip = (len(args.tta_angles) * 2 * args.n_draws
                                * args.n_attach * args.n_rollouts)
        print(f"  T={T:.2f}   recall = {n_hit}/{n_tot} = {n_hit/max(n_tot,1):.3f}   "
              f"({n_rollouts_per_clip} rollouts/clip)", flush=True)


if __name__ == "__main__":
    main()
