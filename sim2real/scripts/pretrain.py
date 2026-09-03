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
    ap.add_argument("--sim", default="flagella",
                    choices=["flagella", "flagella_fast", "many_cells", "many_cells_fast",
                             "many_cells_small", "easy_cells", "two_cells", "multiscale",
                             "worms", "worms_fast", "algae", "algae_fast"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Accumulate gradients over N micro-batches before applying update. "
                         "Effective batch = batch * grad_accum_steps. Enables large effective "
                         "batch when per-step memory is limited (e.g., stride 2 at 128x128).")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=200,
                    help="Linear LR warmup: 0 → lr over this many steps, then cosine decay.")
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
    ap.add_argument("--lambda-aux", type=float, default=0.0)
    ap.add_argument("--lambda-mask-glimpse", type=float, default=0.0)
    ap.add_argument("--lambda-appear-glimpse", type=float, default=0.0,
                    help="Per-slot glimpse-space appearance MSE, foreground-weighted by GT mask. "
                         "Replaces full-composite pixel MSE at pretrain time.")
    ap.add_argument("--lambda-slot-contrast", type=float, default=0.0,
                    help="SlotContrast temporal contrastive on z_what (CVPR 2025).")
    ap.add_argument("--slot-contrast-tau", type=float, default=0.1)
    ap.add_argument("--lambda-slot-repel", type=float, default=0.0,
                    help="Gaussian repulsion penalty between alive slots. Attacks "
                         "'phantom slot in inter-cell region' failure. Try 1.0 or 5.0.")
    ap.add_argument("--slot-repel-sigma", type=float, default=0.1,
                    help="Repulsion kernel sigma in normalized [-1,1] position space. "
                         "0.1 ≈ 3px in 64x64. Should be roughly cell diameter.")
    ap.add_argument("--lambda-pres-smooth", type=float, default=0.0,
                    help="Temporal smoothness on z_pres (MSE between sigmoid(logit) at "
                         "consecutive frames). Sim GT is constant across T, so this attacks "
                         "the pres-flicker / marginal-prediction failure mode directly.")
    ap.add_argument("--pres-focal-gamma", type=float, default=0.0,
                    help="Focal BCE gamma on z_pres. >0 pushes the classifier out of the "
                         "marginal minimum where alive/dead logits overlap. Try 2.0.")
    ap.add_argument("--pres-focal-alpha", type=float, default=0.5,
                    help="Positive-class weight for focal BCE on z_pres. Lower = upweight dead "
                         "slots. With ~83%% positives, try 0.25.")
    ap.add_argument("--match-pres-weight", type=float, default=0.0,
                    help="DETR-style: add -w * p̂_pres(i) to Hungarian cost on alive GT columns. "
                         "Encourages the matcher to send confident-alive predictions to alive GT slots.")
    ap.add_argument("--stop-grad-zwhat", action="store_true",
                    help="Fully decouple recognition from generation: call decoder twice per slot, "
                         "one with stop_grad(zwhat) for the composite so recon can't corrupt zwhat.")
    ap.add_argument("--pres-hidden", type=int, default=128,
                    help="Hidden units in PresHead MLP. Default 128 matches d_model.")
    ap.add_argument("--pres-depth", type=int, default=1,
                    help="Number of hidden MLP layers in PresHead.")
    ap.add_argument("--pres-image-attn", action="store_true",
                    help="Give PresHead direct image access via cross-attention to encoder feature grid.")
    ap.add_argument("--slot-competing-cross", action="store_true",
                    help="Slot Attention-style cross-attention: softmax over slot axis instead of "
                         "key axis, forcing slots to compete for image regions.")
    ap.add_argument("--anchor-slots", action="store_true",
                    help="Replace learnable per-slot embeddings with a fixed grid of sinusoidal-encoded "
                         "anchor positions. Ties slot identity to spatial location.")
    ap.add_argument("--use-slot-attention", action="store_true",
                    help="Use proper iterative Slot Attention (Locatello 2020) instead of DETR-style "
                         "cross-attention. n_transformer_layers reinterpreted as SA iterations.")
    ap.add_argument("--use-neural-em", action="store_true",
                    help="Use Neural EM refiner: structured latents (z_where, z_pres, z_what) as "
                         "iteration state. Replaces slot_transformer + heads.")
    ap.add_argument("--use-isa", action="store_true",
                    help="Use Invariant Slot Attention (Biza et al 2302.04973): equivariant "
                         "per-slot position encoding (pose-rotated + scaled pixel offsets) inside "
                         "the attention. Structured latents like --use-neural-em.")
    ap.add_argument("--isa-abs-pe-weight", type=float, default=0.0,
                    help="Mix in absolute PE into ISA's K_base with this weight (default 0 = "
                         "strict equivariance). Try 0.1 as a fallback when equivariant PE fails.")
    ap.add_argument("--isa-anchor-init", action="store_true",
                    help="Initialize ISA slot positions to a fixed grid (breaks attention<->pose "
                         "chicken-and-egg at t=0).")
    ap.add_argument("--anchor-init-fixed", action="store_true",
                    help="Use a FIXED (non-learned) grid of slot positions as z_where_init. "
                         "Provides consistent cross-video symmetry breaking. Works for both NEM "
                         "and ISA paths.")
    ap.add_argument("--z-what-init-std", type=float, default=0.2,
                    help="Std of the learned z_what_init random init. Wider = more per-slot "
                         "identity discrimination at frame 0. Default 0.2, try 1.0.")
    ap.add_argument("--nem-attn-temp", type=float, default=1.0,
                    help="Softmax temperature in NEM E-step. <1 = sharper responsibility, "
                         "less slot-collapse-to-means. Try 0.5 or 0.25.")
    ap.add_argument("--nem-use-bg-slot", action="store_true",
                    help="Add a phantom 'background' slot to NEM softmax competition. "
                         "Prevents real slots from absorbing background mass (which "
                         "contaminates their centroids). DETR-style null-slot.")
    ap.add_argument("--nem-bg-slot-per-pixel", action="store_true",
                    help="Extend --nem-use-bg-slot: compute the null-slot logit per pixel from a "
                         "small MLP over (image feats, z_style). Lets the bg slot distinguish "
                         "'obviously far-bg pixel' from 'edge of a thin fg object', which the "
                         "scalar version can't.")
    ap.add_argument("--bg-recon-weight", type=float, default=1.0,
                    help="Per-pixel loss weight OUTSIDE the dilated GT fg union. 1.0 = uniform MSE "
                         "(no change). 0.1 = focus 90%% of recon capacity on fg pixels. Useful for "
                         "texture-heavy sims (algae) where bg detail eats slot capacity.")
    ap.add_argument("--bg-recon-dilate-px", type=int, default=3,
                    help="Pixels to dilate the GT fg union before applying --bg-recon-weight. "
                         "Keeps antialiased slot edges in the fg region.")
    ap.add_argument("--nem-dual-source-what", action="store_true",
                    help="Add glimpse_feat (STN-read local image) to z_what before decoding. "
                         "Restores v18 dual-source trick for shape-heavy sims (worms).")
    ap.add_argument("--nem-use-pca-theta", action="store_true",
                    help="Compute NEM slot θ from PCA of responsibility covariance instead "
                         "of tiny residual delta. Essential for arbitrary orientations.")
    ap.add_argument("--decoder-use-film", action="store_true",
                    help="Apply FiLM conditioning on z_what after each ConvT in the glimpse "
                         "decoder. Forces decoder to use z_what content instead of collapsing "
                         "to a mean shape. Helps shape-heavy sims.")
    ap.add_argument("--nem-theta-delta-scale", type=float, default=0.05,
                    help="Separate per-iter delta scale for θ (NEM only). Default 0.05 caps "
                         "θ reach ~27°/frame. Try 0.3 for shape-heavy / arbitrary-orientation "
                         "sims (worms).")
    ap.add_argument("--savi-bootstrap", action="store_true",
                    help="SAVi-style frame-0 conditioning: pass GT z_where[:, 0] as slot init per video, "
                         "removing inter-video permutation randomness on the slot queries.")
    ap.add_argument("--density-curriculum", action="store_true",
                    help="many_cells only: ramp n_objects 10 -> 20 -> 30 -> 40 at steps 0, 25k, 50k, 75k. "
                         "Rebuilds the training batcher at each boundary; eval batcher stays at full density.")
    ap.add_argument("--matching-mode", default="per_frame", choices=["per_frame", "once"],
                    help="Hungarian mode. 'once' matches on frame 0 only and reuses the "
                         "permutation for the whole video (forces temporal identity).")
    ap.add_argument("--match-once-after", type=int, default=0,
                    help="If >0, start with per-frame matching then switch to match-once at "
                         "this step (one JIT recompile). Combines a clean early z_where "
                         "gradient with late-training identity enforcement.")
    ap.add_argument("--hard-pres-gate", action="store_true",
                    help="Forbid dead-pred slots (z_pres≈0) from winning alive-GT columns in "
                         "the Hungarian matcher (train side). Adds a 1e6 penalty on "
                         "(1-pred_pres)*gt_pres pairings. Fixes the mechanism where a dead "
                         "slot's z_where drifts near a real cell and steals its assignment.")
    ap.add_argument("--glimpse-size", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--z-what-dim", type=int, default=64,
                    help="Latent z_what dimensionality per slot. Bump for shape-heavy sims "
                         "where slots need to encode internal per-element structure.")
    ap.add_argument("--stem-channels", type=int, nargs="+", default=[16, 32, 64],
                    help="Per-stage CNN channel counts in the encoder stem. Bigger = more capacity.")
    ap.add_argument("--n-transformer-layers", type=int, default=2)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 2],
                    help="Per-stage CNN strides. Default (2,2,2) gives total stride 8 → 16x16 grid. "
                         "Use (2,2,1) for stride 4 → 32x32 grid (halved pixel quantum).")
    ap.add_argument("--encoder-type", default="vit", choices=["vit", "unet"],
                    help="Encoder backbone. 'vit' = ConvStem+ViT (uses --stem-channels/--stem-strides). "
                         "'unet' = down/up with skip concats, no self-attention (O(H*W*d) memory).")
    ap.add_argument("--unet-down-channels", type=int, nargs="+",
                    default=[24, 48, 96, 128, 192],
                    help="U-Net down-path channel widths. Length K gives K stages, resolutions "
                         "128, 64, ..., 128/2^(K-1). Default → 128,64,32,16,8.")
    ap.add_argument("--unet-n-up-stages", type=int, default=2,
                    help="U-Net upsampling steps after the bottleneck. 2 → 32x32 grid, 3 → 64x64.")
    ap.add_argument("--unet-blocks-per-stage", type=int, default=2,
                    help="Conv→GN→GELU blocks per U-Net stage.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Held-out eval every N steps (0 = disabled). Logs eval/{recon_mse,psnr,ssim,seg_iou,silhouette_zwhat}.")
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=424242)
    ap.add_argument("--sim-t", type=int, default=None,
                    help="Override sim.common.T (sequence length). Cheaper training at T=2.")
    ap.add_argument("--algae-single-flag-prob", type=float, default=None,
                    help="Override algae's single_flag_prob. Set 0.0 to guarantee 3 objects per video.")
    ap.add_argument("--algae-pipette-prob", type=float, default=None,
                    help="Override algae's pipette_prob. Set 1.0 to always render + supervise pipette (slot 3).")
    ap.add_argument("--algae-body-rx-min", type=float, default=None,
                    help="Override body_rx_min. Widen to smaller cells (e.g. 0.08).")
    ap.add_argument("--algae-body-dark-min", type=float, default=None,
                    help="Override body_dark_intensity_min. Lower = fainter cells (e.g. 0.10).")
    ap.add_argument("--algae-img-blur-max", type=float, default=None,
                    help="Override img_blur_sigma_max. Higher = more defocus (e.g. 2.5).")
    args = ap.parse_args()

    # Pick a reasonable model n_max per sim (must be ≤ sim n_max).
    default_n_max = {"flagella": 8, "flagella_fast": 6, "many_cells": 48, "many_cells_fast": 24,
                     "many_cells_small": 48, "easy_cells": 8, "two_cells": 4,
                     "multiscale": 16, "worms": 12, "worms_fast": 8,
                     "algae": 6, "algae_fast": 6}
    n_max = args.n_max if args.n_max is not None else default_n_max[args.sim]

    model_cfg = ModelConfig(
        n_max=n_max,
        d_model=args.d_model,
        n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=args.z_what_dim,
        z_style_dim=4,
        glimpse_size=args.glimpse_size,
        stem_channels=tuple(args.stem_channels),
        stem_strides=tuple(args.stem_strides),
        n_groups=args.n_groups,
        use_background=True,
        bg_base_res=4,
        bg_channels=(8,),
        stop_grad_recon_zwhat=args.stop_grad_zwhat,
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
        anchor_init_fixed=args.anchor_init_fixed,
        z_what_init_std=args.z_what_init_std,
        nem_attn_temp=args.nem_attn_temp,
        nem_use_bg_slot=args.nem_use_bg_slot,
        nem_bg_slot_per_pixel=args.nem_bg_slot_per_pixel,
        nem_dual_source_what=args.nem_dual_source_what,
        nem_use_pca_theta=args.nem_use_pca_theta,
        decoder_use_film=args.decoder_use_film,
        nem_theta_delta_scale=args.nem_theta_delta_scale,
        encoder_type=args.encoder_type,
        unet_down_channels=tuple(args.unet_down_channels),
        unet_n_up_stages=args.unet_n_up_stages,
        unet_blocks_per_stage=args.unet_blocks_per_stage,
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
        lambda_aux=args.lambda_aux,
        lambda_mask_glimpse=args.lambda_mask_glimpse,
        lambda_appear_glimpse=args.lambda_appear_glimpse,
        lambda_slot_contrast=args.lambda_slot_contrast,
        slot_contrast_tau=args.slot_contrast_tau,
        lambda_slot_repel=args.lambda_slot_repel,
        slot_repel_sigma=args.slot_repel_sigma,
        lambda_pres_smooth=args.lambda_pres_smooth,
        pres_focal_gamma=args.pres_focal_gamma,
        pres_focal_alpha=args.pres_focal_alpha,
        match_pres_weight=args.match_pres_weight,
        matching_mode=args.matching_mode,
        hard_pres_gate=args.hard_pres_gate,
        bg_recon_weight=args.bg_recon_weight,
        bg_recon_dilate_px=args.bg_recon_dilate_px,
    )
    # Optional sim-config overrides (only applied when the corresponding CLI flag is passed).
    sim_cfg_override = None
    _algae_overrides = [
        args.algae_single_flag_prob, args.algae_pipette_prob,
        args.algae_body_rx_min, args.algae_body_dark_min, args.algae_img_blur_max,
    ]
    if args.sim_t is not None or any(x is not None for x in _algae_overrides):
        import dataclasses as _dc
        from sim2real.sim.api import _REGISTRY
        default_cfg_cls = _REGISTRY[args.sim][2]
        sim_cfg_override = default_cfg_cls()
        if args.sim_t is not None:
            sim_cfg_override = _dc.replace(
                sim_cfg_override,
                common=_dc.replace(sim_cfg_override.common, T=args.sim_t),
            )
        if args.algae_single_flag_prob is not None:
            if args.sim not in ("algae", "algae_fast"):
                raise ValueError("--algae-single-flag-prob only valid with --sim algae/algae_fast")
            sim_cfg_override = _dc.replace(
                sim_cfg_override, single_flag_prob=args.algae_single_flag_prob,
            )
        if args.algae_pipette_prob is not None:
            if args.sim not in ("algae", "algae_fast"):
                raise ValueError("--algae-pipette-prob only valid with --sim algae/algae_fast")
            sim_cfg_override = _dc.replace(
                sim_cfg_override, pipette_prob=args.algae_pipette_prob,
            )
        if args.algae_body_rx_min is not None:
            if args.sim not in ("algae", "algae_fast"):
                raise ValueError("--algae-body-rx-min only valid with algae")
            sim_cfg_override = _dc.replace(
                sim_cfg_override, body_rx_min=args.algae_body_rx_min,
                body_ry_min=args.algae_body_rx_min,  # tie ry to rx for isotropic small cells
            )
        if args.algae_body_dark_min is not None:
            if args.sim not in ("algae", "algae_fast"):
                raise ValueError("--algae-body-dark-min only valid with algae")
            sim_cfg_override = _dc.replace(
                sim_cfg_override, body_dark_intensity_min=args.algae_body_dark_min,
            )
        if args.algae_img_blur_max is not None:
            if args.sim not in ("algae", "algae_fast"):
                raise ValueError("--algae-img-blur-max only valid with algae")
            sim_cfg_override = _dc.replace(
                sim_cfg_override, img_blur_sigma_max=args.algae_img_blur_max,
            )
        print(f"[sim-override] sim_cfg={sim_cfg_override}", flush=True)

    cfg = PretrainConfig(
        sim_kind=args.sim,
        sim_cfg=sim_cfg_override,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum_steps,
        n_steps=args.steps,
        lr_peak=args.lr,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        run_dir=args.run_dir or f"runs/pretrain_{args.sim}",
        seed=args.seed,
        teacher_force_zwhere=args.teacher_force_zwhere,
        teacher_force_zpres=args.teacher_force_zpres,
        t_curriculum=args.t_curriculum,
        t_start=args.t_start,
        t_curriculum_steps=args.t_curriculum_steps,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch,
        eval_seed=args.eval_seed,
        match_once_after=args.match_once_after,
        density_curriculum=args.density_curriculum,
        savi_bootstrap=args.savi_bootstrap,
    )
    train_pretrain(cfg)


if __name__ == "__main__":
    main()
