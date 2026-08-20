"""Entrypoint for unsupervised adaptation.

Load a pretrained checkpoint and adapt it to a shifted simulator using recon+KL only
(no Hungarian, no matching, no supervised terms). By default freezes the generative
side (`glimpse_decoder`, `bg_renderer`) and trains only the inference network — this is
the "adapt just the latents" recipe.

Example:
    PYTHONPATH=. python3 -m sim2real.scripts.adapt \\
        --pretrain-ckpt runs/slotattn_manycells_200k/ckpts/step_200000.pkl \\
        --sim many_cells_small --n-max 48 \\
        --glimpse-size 24 --n-transformer-layers 3 --stem-strides 2 2 1 \\
        --use-slot-attention \\
        --steps 5000 --run-dir runs/adapt_slotattn_small
"""

from __future__ import annotations

import argparse

from sim2real.losses.losses import AdaptLossConfig
from sim2real.model.model import ModelConfig
from sim2real.train.adapt import AdaptConfig, train_adapt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="many_cells")
    ap.add_argument("--pretrain-ckpt", default="")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--freeze", nargs="*",
                    default=["glimpse_decoder.*", "bg_renderer.*"])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--kl-ramp-steps", type=int, default=500)
    # Loss weights (override AdaptLossConfig defaults).
    ap.add_argument("--lambda-recon", type=float, default=1.0)
    ap.add_argument("--lambda-kl", type=float, default=0.05)
    ap.add_argument("--lambda-alive", type=float, default=0.0,
                    help="Anti-collapse hinge on mean(z_pres). Prevents the STN opt-out shortcut.")
    ap.add_argument("--target-alive-rate", type=float, default=0.5,
                    help="Target for mean(z_pres). Ideally ~ n_expected_cells / n_max.")
    ap.add_argument("--use-feat-recon", action="store_true",
                    help="DINOSAUR-lite: replace pixel MSE with per-pixel feature MSE against a "
                         "frozen target-encoder snapshot. Prevents STN opt-out collapse.")
    ap.add_argument("--lambda-feat-recon", type=float, default=1.0)
    # Model config (must match pretrained ckpt).
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=3)
    ap.add_argument("--z-what-dim", type=int, default=64)
    ap.add_argument("--z-style-dim", type=int, default=4)
    ap.add_argument("--glimpse-size", type=int, default=24)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--n-groups", type=int, default=1)
    ap.add_argument("--bg-base-res", type=int, default=4)
    ap.add_argument("--bg-channels", type=int, nargs="+", default=[8])
    ap.add_argument("--pres-hidden", type=int, default=128)
    ap.add_argument("--pres-depth", type=int, default=1)
    ap.add_argument("--pres-image-attn", action="store_true")
    ap.add_argument("--slot-competing-cross", action="store_true")
    ap.add_argument("--anchor-slots", action="store_true")
    ap.add_argument("--use-slot-attention", action="store_true")
    ap.add_argument("--use-neural-em", action="store_true")
    ap.add_argument("--use-isa", action="store_true")
    ap.add_argument("--isa-abs-pe-weight", type=float, default=0.0)
    ap.add_argument("--isa-anchor-init", action="store_true")
    ap.add_argument("--savi-bootstrap", action="store_true",
                    help="Pass GT z_where[:, 0] as slot init per video. Set only if the "
                         "pretrained model expects it. Leaks frame-0 GT into adaptation.")
    args = ap.parse_args()

    default_n_max = {"flagella": 8, "many_cells": 48, "many_cells_small": 48,
                     "many_cells_fast": 24, "many_cells_fast_small": 24,
                     "easy_cells": 8, "two_cells": 4, "multiscale": 16, "worms": 12}
    n_max = args.n_max if args.n_max is not None else default_n_max[args.sim]

    model_cfg = ModelConfig(
        n_max=n_max,
        d_model=args.d_model,
        n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=args.z_what_dim,
        z_style_dim=args.z_style_dim,
        glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64),
        stem_strides=tuple(args.stem_strides),
        n_groups=args.n_groups,
        use_background=True,
        bg_base_res=args.bg_base_res,
        bg_channels=tuple(args.bg_channels),
        pres_hidden=args.pres_hidden,
        pres_depth=args.pres_depth,
        pres_image_attn=args.pres_image_attn,
        slot_competing_cross=args.slot_competing_cross,
        anchor_slots=args.anchor_slots,
        use_slot_attention=args.use_slot_attention,
        use_neural_em=args.use_neural_em,
        use_isa=args.use_isa,
        isa_abs_pe_weight=args.isa_abs_pe_weight,
        isa_anchor_init=args.isa_anchor_init,
    )
    loss_cfg = AdaptLossConfig(lambda_recon=args.lambda_recon, lambda_kl=args.lambda_kl,
                                lambda_alive=args.lambda_alive,
                                target_alive_rate=args.target_alive_rate)
    cfg = AdaptConfig(
        sim_kind=args.sim,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        batch_size=args.batch,
        n_steps=args.steps,
        lr_peak=args.lr,
        warmup_steps=args.warmup_steps,
        kl_ramp_steps=args.kl_ramp_steps,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        freeze_patterns=tuple(args.freeze),
        pretrain_ckpt=args.pretrain_ckpt,
        run_dir=args.run_dir or f"runs/adapt_{args.sim}",
        seed=args.seed,
        savi_bootstrap=args.savi_bootstrap,
        use_feat_recon=args.use_feat_recon,
        lambda_feat_recon=args.lambda_feat_recon,
    )
    train_adapt(cfg)


if __name__ == "__main__":
    main()
