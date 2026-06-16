# sim2real_tracking

Object-centric video model that disentangles per-object latents (`z_what`, `z_where`, `z_pres`) and a per-video latent (`z_style`), trained on synthetic microscopy-style videos and adapted across prior settings or to real data.

## Research question

Can the model be **pretrained supervised** on simulator A and then **adapted unsupervised** (reconstruction loss only) to simulator B with different prior settings, such that the latents retain meaningful semantics? Later: real microscopy.

## Components

- **Four simulators** (`sim2real/sim/`)
  1. Flagella-like — few elongated objects, internal beating motion (algae lineage).
  2. Many similar cells — random walks of small circular objects (cells lineage).
  3. Multi-scale — mixed large nuclei + small puncta.
  4. Same-shape worms — shared learned template, varied pose.
- **Slot-based encoder/decoder** (`sim2real/model/`) — DETR-style propagate-and-discover with per-slot 1-step GRU. Decoder has a separate segmentation head.
- **Priors** (`sim2real/priors/`) — z_where random walk, z_pres birth/death, z_what AR(1) + cross-object KL, z_style i.i.d.
- **Pretrain + adapt loops** (`sim2real/train/`) — supervised on A, unsupervised on B via `optax.masked` freeze.
- **Eval** (`sim2real/eval/`) — recon, seg IoU, tracking, latent-disentanglement probes.

## Entrypoints

```bash
python -m sim2real.scripts.render_sim_smoketest --sim flagella_A --out runs/sim_smoke
python -m sim2real.scripts.pretrain  --cfg configs/experiment/pretrain_flagella_A.yaml
python -m sim2real.scripts.adapt     --cfg configs/experiment/adapt_flagella_A_to_B.yaml
python -m sim2real.scripts.eval_ckpt --ckpt runs/<exp>/ckpts/last
python -m sim2real.scripts.sweep_sim2real --pretrained runs/pretrain_A/ckpts/last
```

## Plan

See `/home/frans/.claude/plans/i-m-starting-a-new-purring-lemon.md`.
