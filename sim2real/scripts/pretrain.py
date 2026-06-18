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
    ap.add_argument("--teacher-force-zwhere", action="store_true")
    ap.add_argument("--teacher-force-zpres", action="store_true")
    ap.add_argument("--t-curriculum", action="store_true")
    ap.add_argument("--t-start", type=int, default=3)
    ap.add_argument("--t-curriculum-steps", type=int, default=25_000)
    ap.add_argument("--n-groups", type=int, default=1)
    ap.add_argument("--lambda-group", type=float, default=0.0)
    ap.add_argument("--lambda-group-temp", type=float, default=0.0)
    ap.add_argument("--glimpse-size", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=2)
    args = ap.parse_args()

    # Pick a reasonable model n_max per sim (must be ≤ sim n_max).
    default_n_max = {"flagella": 8, "many_cells": 48, "multiscale": 16, "worms": 12}
    n_max = args.n_max if args.n_max is not None else default_n_max[args.sim]

    model_cfg = ModelConfig(
        n_max=n_max,
        d_model=args.d_model,
        n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=64,
        z_style_dim=4,
        glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64),
        n_groups=args.n_groups,
        use_background=True,
        bg_base_res=4,
        bg_channels=(8,),
    )
    from sim2real.losses.losses import PretrainLossConfig
    loss_cfg = PretrainLossConfig(
        lambda_recon=args.lambda_recon,
        lambda_where=args.lambda_where,
        lambda_pres=args.lambda_pres,
        lambda_mask=args.lambda_mask,
        lambda_kl=0.0,
        lambda_group=args.lambda_group,
        lambda_group_temp=args.lambda_group_temp,
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
        teacher_force_zwhere=args.teacher_force_zwhere,
        teacher_force_zpres=args.teacher_force_zpres,
        t_curriculum=args.t_curriculum,
        t_start=args.t_start,
        t_curriculum_steps=args.t_curriculum_steps,
    )
    train_pretrain(cfg)


if __name__ == "__main__":
    main()
