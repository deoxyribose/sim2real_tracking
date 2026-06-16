"""Entrypoint for supervised pretraining.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.pretrain --sim many_cells --steps 500
"""

from __future__ import annotations

import argparse

from sim2real.model.model import ModelConfig
from sim2real.train.pretrain import PretrainConfig, train_pretrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="flagella", choices=["flagella", "many_cells", "multiscale", "worms"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda-recon", type=float, default=1.0)
    ap.add_argument("--lambda-where", type=float, default=1.0)
    ap.add_argument("--lambda-pres", type=float, default=0.5)
    ap.add_argument("--lambda-mask", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=500)
    args = ap.parse_args()

    # Pick a reasonable model n_max per sim (must be ≤ sim n_max).
    default_n_max = {"flagella": 8, "many_cells": 48, "multiscale": 16, "worms": 12}
    n_max = args.n_max if args.n_max is not None else default_n_max[args.sim]

    model_cfg = ModelConfig(
        n_max=n_max,
        d_model=128,
        n_heads=4,
        n_transformer_layers=2,
        z_what_dim=64,
        z_style_dim=16,
        glimpse_size=16,
        stem_channels=(16, 32, 64),
    )
    from sim2real.losses.losses import PretrainLossConfig
    loss_cfg = PretrainLossConfig(
        lambda_recon=args.lambda_recon,
        lambda_where=args.lambda_where,
        lambda_pres=args.lambda_pres,
        lambda_mask=args.lambda_mask,
        lambda_kl=0.0,
    )
    cfg = PretrainConfig(
        sim_kind=args.sim,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        batch_size=args.batch,
        n_steps=args.steps,
        lr_peak=args.lr,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        run_dir=args.run_dir or f"runs/pretrain_{args.sim}",
        seed=args.seed,
    )
    train_pretrain(cfg)


if __name__ == "__main__":
    main()
