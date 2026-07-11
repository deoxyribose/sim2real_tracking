# sim2real_tracking — experiments log

Running log of what we've tried and what we learned. Newest at the top.

**Held-out eval metrics** are computed by the in-loop eval hook on a fixed 4-video batch sampled at `eval_seed=424242`. Metrics: pixel MSE, PSNR, SSIM, matched IoU (Hungarian on `z_where`), and silhouette score on `z_what` clustered by GT slot identity. Silhouette ranges [-1, 1] — negative means slots for different cells are closer in `z_what` space than slots for the same cell (anti-clustered).

**Historical baseline** (`runs/unified_many_cells_v2`, 100k steps, June 2026): PSNR 24.16, SSIM 0.52, IoU 0.127, silhouette -0.034, id_switches 363.

---

## 2026-07-02 — session: fix pretrain decomposition

### Exp 14: `long_all_sims` — **Beats the June baseline on every sim** (2026-07-10 → 2026-07-11, 23.7h wall)

200k steps × 6 configs. The Exp 13 recipe on all four sims, with two extra n_groups=1 variants for flagella and worms to ablate the grouped decoder against SlotContrast + match-once.

| sim | our IoU | historical (100k) | × | our silhouette | historical | our PSNR |
|-----|---------|-------------------|---|-----------------|------------|----------|
| flagella_grp1 | **0.187** | 0.078 | **2.4×** | 0.472 | 0.028 | 19.64 |
| flagella_grp8 | 0.183 | 0.078 | 2.3× | 0.499 | 0.028 | 19.68 |
| worms_grp1 | **0.228** | 0.129 | **1.8×** | 0.578 | -0.015 | 21.19 |
| worms_grp12 | 0.228 | 0.129 | 1.8× | 0.576 | -0.015 | 21.45 |
| multiscale | 0.062 | 0.018 | **3.4×** | 0.374 | -0.090 | 22.87 |
| many_cells | 0.184 | 0.127 | 1.4× | **0.743** | -0.034 | 22.98 |

**Every sim above the June baseline on IoU (1.4–3.4×). Silhouette flipped from negative/zero to strongly positive on every sim.**

**n_groups ablation:**
- **flagella**: groups helped identity (id_switches 42 → **19**, -55%) but did not change IoU or silhouette.
- **worms**: groups changed nothing measurably (both IoU 0.228, both silhouette 0.58).
- Interpretation: groups matter when instances look alike (thin filaments) but are redundant when each instance is visually distinctive.

**many_cells** is the smallest gain (1.4×) and the only sim where PSNR/SSIM regressed vs historical (22.98/0.30 vs 24.16/0.52). This is the densest, most homogeneous sim — the composite pixel-MSE we replaced with glimpse-space per-slot losses may have been contributing more here than elsewhere. Worth revisiting the recon weighting for many_cells specifically.

**Config:** stride-4 encoder (32×32 grid), matching-mode curriculum (per_frame → once at step 500), no teacher-force-zpres, no SAVi bootstrap, scale-init bias, lambda_where=5, SlotContrast + glimpse-space appearance + glimpse-space mask + composite recon. Full script at `scripts/long_all_sims.sh`. Summary plot at `runs/long_all_sims/summary.png`.

---

### Exp 13: `stride4` — **IoU 0.115 in 18 min via stride-4 encoder (32×32 grid)** (2026-07-10)

Halve the encoder's total stride (2,2,2 → 2,2,1) so the final feature grid is 32×32 instead of 16×16. Halves the pixel quantum from 8 → 4 px/token, matching SLATE/STEVE/SlotFormer. No other changes.

**Files touched:**
- `sim2real/model/encoder.py`: added `strides` param to `ConvStem` and `FrameEncoder` (default (2,2,2) preserves prior behavior).
- `sim2real/model/model.py`: `ModelConfig.stem_strides` field.
- `sim2real/scripts/pretrain.py`: `--stem-strides` CLI flag.

**Results (10k steps, 17.7 min wall time — barely slower than Exp 12):**
| step  | Exp 12 IoU | **Exp 13 IoU** | Exp 12 sil | Exp 13 sil |
|-------|-----------|-----------|-----------|-----------|
| 1000  | 0.040 | 0.045 | 0.721 | 0.767 |
| 3000  | 0.068 | 0.078 | 0.781 | 0.750 |
| 5000  | 0.082 | 0.093 | 0.744 | 0.750 |
| 7500  | —     | 0.111 | —     | 0.746 |
| **10000** | **0.105** | **0.115** | 0.692 | 0.743 |

**Position error dropped 42%** (0.137 → 0.079 in tanh space, ~9 px → ~5 px — at the stride-4 quantum floor). Scale stayed accurate (pred 0.107 vs GT 0.109). Train `L_where` dropped 15% (0.117 → 0.099).

**Wall-time surprise:** theoretical 4× cross-attention cost translated to only +8% wall time. The GPU was 2% utilized at 16×16; adding 4× tokens uses the headroom, not more clock time.

**Vs historical baseline (100k steps, IoU 0.127):** now at 91% of historical performance in 10% of the compute, still monotonically climbing at step 10000.

---

### Exp 12: `scale_fix_jax010` — **JAX 0.5.0 → 0.10.2 gives 100× speedup + IoU 0.105 in 16 min** (2026-07-10)

Rebuilt venv at `/home/frans/jaxup_venv` with JAX 0.10.2 (was JAX 0.5.0, installed Feb 2025). Same config as Exp 11.

**Speedup:** 1663 ms/step → 16.2 ms/step (**103× faster**). Historical June baseline was ~23 ms/step; we're now faster than that. The whole "70× slowdown vs. historical" throughput regression was **stale JAX** (0.5.0 is over a year old and lacks proper Blackwell 5090 dispatch).

**Bonus:** T=8 CUDA_ERROR_ILLEGAL_ADDRESS crash that broke Exps 5, 7, 8 is *also gone*. Run reached T=12 (native) with no error. That was another JAX 0.5.0 bug.

**Results (10k steps, 16.4 min wall time):**
| step  | eval IoU | silhouette | PSNR  | train where |
|-------|----------|------------|-------|-------------|
| 1     | 0.029    | -0.035     | 20.28 | 0.47        |
| 500   | 0.039    | 0.543      | 21.62 | —           |
| 1000  | 0.040    | 0.721      | 21.65 | 0.23        |
| 3000  | 0.068    | 0.781      | 21.80 | —           |
| 5000  | 0.085    | 0.744      | 21.84 | —           |
| **10000** | **0.105** | **0.692** | **21.91** | **0.12** |

**IoU 0.105 at step 10000 vs historical 0.127 at step 100000** — 82% of historical converged IoU in 10% of the compute, and still climbing. If we run to 100k (~35 min), likely exceed historical.

**Files touched:** `scripts/rebalance_many_cells.sh` (PY=/home/frans/jaxup_venv/bin/python, STEPS=10000). NaN-guard was also rewritten in `sim2real/train/pretrain.py` (gnorm-based instead of tree.map + lax.cond) but this change contributes negligibly compared to the JAX upgrade.

---

### Exp 11: `scale_fix` — scale-init bias + `lambda-where 5` (2026-07-10, killed at step 1600 for JAX upgrade)

Fixed the scale problem identified by Exp 10.5 diagnostic. `z_where_init` scale channels re-biased near sigmoid 0.10 (empirical GT mean) via a `-2.2` bias. `--lambda-where 1.0 → 5.0`.

**Confirmed scale is now correct at step 1000:**
- pred sx/sy mean 0.10 vs GT 0.10 — exact match
- Position error 0.137 (unchanged) — position still ~9 pixels off from cells

**Broke the 0.02 IoU ceiling** that had held for Exps 3-10: reached IoU 0.041 at step 1500. Silhouette climbed to 0.79 without needing the curriculum switch (per-frame Hungarian with proper scale init gave SlotContrast enough traction).

---

### Exp 10.5: mask + z_where diagnostic on Exp 10 step-1600 ckpt

Killed the "diffuse blob" hypothesis. Per-slot masks are already **sharp compact dots** at distinct locations, `p=1.00`. The illusion of a "diffuse blob" in the summary panel was just 45 sharp dots summed.

Actual problem: **pred scale is 3× too large** (pred sx 0.33 vs GT 0.10), so each sharp dot is inside a mostly-empty large STN write area. And **position error is ~9 pixels** (0.137 in tanh space) ~ a full cell diameter — pred dots are close to cells but often not overlapping. Motivated Exp 11's scale-init fix.

---

### Exp 10: `match_curriculum` (2026-07-03, **partial success — silhouette flipped, IoU stuck**)

Per-frame Hungarian for the first 500 steps (let z_where descend on the best per-frame supervised signal), then rebuild `train_step` with `matching_mode='once'` to lock in temporal identity for the remaining 1500 steps. One JIT recompile at the transition.

Motivation: Exp 9 (match_once from step 1) hurt short-term z_where descent because Hungarian at random-init z_where gives a bad frame-0 assignment that gets enforced at every frame. Curriculum lets the model first find approximate slot locations under permissive per-frame matching, then enforces temporal identity once matching is meaningful.

**Files touched:**
- `sim2real/train/pretrain.py`: `PretrainConfig.match_once_after: int = 0`; rebuild `train_step` at that step by `dataclasses.replace(loss_cfg, matching_mode='once')`.
- `sim2real/scripts/pretrain.py`: `--match-once-after N` CLI flag.

**Config:** All Exp 8b config (recon+appearG+mask_glimpse+slotcontrast, no bootstrap, no teacher-force-zpres, T-curriculum-steps=5000) + `--matching-mode per_frame --match-once-after 500`.

**Results so far:**
| step | contr | where | eval IoU | eval silhouette |
|------|-------|-------|----------|-----------------|
| 500 (pre-switch)  | 3.87  | 2.96 | 0.020    | -0.223 |
| 750 (post-switch) | —     | —    | 0.017    | -0.049 |
| **1000 (post)**   | **0.072** | **1.94** | 0.020 | **+0.611** |

**Curriculum matching worked.** `contr` collapsed 53× (uniform → 0.072) in 500 post-switch steps. Silhouette flipped from -0.223 to **+0.611**. Temporal identity is now enforced; z_what strongly clusters by GT identity.

**But IoU is stuck at 0.020.** Even though z_where is descending (2.96 → 1.94), IoU has been flat since step 500. Diagnosis: masks are still diffuse blobs — the mask decoder isn't producing sharp per-cell shapes. `train mask` metric plateaued at 0.95 for hundreds of steps. Options: (1) longer training, (2) mask entropy penalty, (3) bigger decoder.

---

### Exp 9: `match_once` (2026-07-03, killed after step 500 for Exp 10)

Match-once from step 1 hurt: at step 500, `train where 2.98` vs Exp 8's 2.05, IoU 0.018 vs 0.029. `contr` stuck at 3.87. Reason: Hungarian at random-init z_where gives bad frame-0 permutation → enforced across all frames → noisy gradient on model's z_where. Motivated Exp 10's curriculum.

Add match-once-use-forever: Hungarian at frame 0 only, permutation broadcast to all T frames. Forces slot n to represent GT slot k(n) for the whole video. Removes the temporal-identity slack that Hungarian-per-frame allowed.

Motivation: Exp 8/8b showed z_where is honestly learning without bootstrap, but silhouette stays negative because slot identity across time is unconstrained by the supervised losses. Per-frame Hungarian lets slot n represent different cells at different t; the supervised losses are indifferent to which cell slot n represents at each frame. With match-once, supervised losses on frames t≥1 directly enforce "keep tracking the same cell as frame 0".

**Files touched:**
- `sim2real/losses/losses.py`: `PretrainLossConfig.matching_mode: str = "per_frame"`; `_match_video(out, sample, mode)` — mode "once" runs Hungarian on frame 0 only and broadcasts.
- `sim2real/scripts/pretrain.py`: `--matching-mode {per_frame,once}` CLI flag.

**Config:** All Exp 8b config + `--matching-mode once`.

**Expectation:** silhouette should go positive (temporal identity now enforced by loss + architecture). IoU trajectory should be similar to Exp 8 (z_where still learned from scratch) but ideally with tighter cross-frame consistency.

---

### Exp 8b: `no_bootstrap_2` (2026-07-03, killed after step 300 for Exp 9)

Same as Exp 8 but with `t_curriculum_steps 1000 → 5000` to keep T ≤ 6 across the 2000-step run and avoid the T=8 CUDA crash. Confirmed Exp 8 trajectory at step 300 (train `where 3.06`, similar to Exp 8's at same step).

---

### Exp 8: `no_bootstrap` (2026-07-03, **crashed after step 600**)

Drop SAVi frame-0 bootstrap entirely. Model has to learn `z_where_init` and residual-head dynamics from scratch. Otherwise identical to Exp 7 (no teacher-force-zpres either).

Motivation: Exp 7.5 z_where inspection showed a third train/eval mismatch: bootstrap replaces frame-0 anchor with GT during training, so `z_where_init` param never gets a gradient. At eval (no bootstrap), model falls back to `z_where_init` (random) and residual head passes it through unchanged (frame 0 == frame 1 == frame 2 for every slot). z_where is essentially not learned.

**Files touched:** `sim2real/train/pretrain.py` (removed bootstrap pass-through in `loss_fn`). `bootstrap_zwhere0` arg still exists in `SlotVideoModel.__call__` for future use.

**Config:** All Exp 7 config + no bootstrap. **ckpt-every 200** (was 500) — Exp 5 and Exp 7 both crashed with `CUDA_ERROR_ILLEGAL_ADDRESS` shortly after a T-curriculum bump around step 500-600. Want to preserve state more often.

**Results so far:**
| step | eval IoU | silhouette | train where |
|------|----------|------------|-------------|
| 1    | 0.013    | -0.035     | 9.29        |
| 250  | 0.014    | -0.171     | 2.63        |
| 500  | **0.029** | -0.288     | 2.05        |

**Crashed after step 600 with `CUDA_ERROR_ILLEGAL_ADDRESS`** — same failure as Exp 5 and Exp 7. Pattern is now confirmed: all three crashes occurred **shortly after a T-curriculum bump into T=8** (t_curriculum-steps=1000 with t_start=3 → T=8 lands around step 555). This is a JAX/XLA/CUDA bug specific to the graph at T=8 with n_max=48 batch=4 — not a training-loop bug. Workaround in Exp 8b: slow the curriculum so T never exceeds 7 in-run.

Last logged step before crash: step 600 T=8, train `where 1.33` — a 35% drop in 100 steps, so z_where descent was actually accelerating right before the crash. `step_600.pkl` saved.

**IoU broke the ~0.023 ceiling** that had held across every Exp 3-7 config. First honest signal of z_where actually being learned. Silhouette continues to fall (slot identity unstable across time — SlotContrast can't get traction on shifting slot-to-cell assignments), but z_where localization is genuinely descending.

Wall time: ~4 s/step, celegans job intermittent.

---

### Exp 7.5: z_where diagnostic on Exp 7 step-500 ckpt

**Finding:** every predicted per-slot mask is a diffuse blob at image center; `z_pres` is now correct (40 alive matching GT); silhouette very positive (+0.891) — but IoU stuck at 0.024.

`z_where` at eval for slot 0: frame 0 (0.45, -0.31), frame 1 (0.45, -0.31), frame 2 (0.45, -0.31), … — identical across all frames. GT for the same slot: frame 0 (-0.59, 0.67), frame 1 (-0.61, 0.62). Model's z_where is stuck at whatever it was on frame 0 and doesn't respond to input.

**Root cause: another train/eval mismatch.**
- Training: SAVi bootstrap replaces frame-0 anchor with GT z_where[0] → `z_where_init` param never receives gradient → stays at its `normal * 0.5` init values → random per-slot positions.
- Eval: no bootstrap → model uses (random) `z_where_init` for frame 0 → residual head produces delta ≈ 0 (zero-init last layer never learned to move slots because bootstrap covered its job during training) → frame t just copies frame 0.

Effectively: `z_pres` and `z_what` are learned, but `z_where` is essentially not.

**Files touched:** none. Diagnostic only.

**Motivation for Exp 8:** drop SAVi bootstrap.

---

### Exp 7: `no_teacher_zpres` (2026-07-03, **crashed after step 500**)

Drop `--teacher-force-zpres`. Otherwise identical to Exp 6.

Motivation: Exp 6.5 diagnostic proved z_pres was never actually learned self-consistently. All Exp 3-6 measured a broken eval forward. Must let the pres_head learn from step 1 without the training gate cheat.

**Config:** many_cells, batch 4, n_max=48, T-curriculum 3→12 over 1000 steps. `--lambda-recon 2.0 --lambda-mask-glimpse 10.0 --lambda-appear-glimpse 10.0 --lambda-slot-contrast 1.0 --lambda-where 1.0 --lambda-pres 1.0`. **No teacher forcing.** 2000 steps, ckpt-every 500, eval every 250.

**Results so far:**
| step | eval IoU | silhouette | PSNR | notes |
|------|----------|------------|------|-------|
| 1    | 0.013    | -0.035     | 20.29 | init only |
| 250  | 0.021    | **+0.746** | 21.76 | silhouette sign-flipped |
| 500  | 0.024    | **+0.891** | 21.44 | silhouette climbing toward 1.0 |

Train `L_pres` descended 4.34 → 0.12 by step 500 — the pres head is *actually* learning self-consistently now. Train `contr` at 0.013 (well minimized).

**Comparison at step 500 eval (vs Exp 6):**
- IoU: 0.024 vs 0.023 — basically same
- silhouette: **+0.891 vs -0.058** — flipped sign, near-perfect z_what clustering

**Sign-flipped silhouette** is the headline: z_what now strongly clusters by GT identity. Confirms Exp 6.5 diagnosis: teacher-force-zpres was making eval-time z_what look like noise. **But IoU is still stuck at ~0.024** — that's the mask/where quality problem, unrelated to z_pres or z_what specialization.

Next check: does IoU catch up as training continues, or is there a mask ceiling? If plateau, do mask viz on step_500.pkl.

---

### Exp 6.5: `rebalance_slotcontrast` mask diagnostic

Inspected `runs/rebalance_slotcontrast/ckpts/step_500.pkl`. Rendered:
- **Frame summary**: GT frame with 40 discrete cells vs. predicted mask sum showing one diffuse red blob covering the center
- **Per-slot masks**: every one of 24 top-p predicted slots looks like an identical diffuse blob; every `z_pres = 0.00`

Actual `z_pres_logit` stats at eval on frame 0: **min -9.06, max -4.98, mean -6.65**. Every single logit is deeply negative → sigmoid(-5) ≈ 0.007 → no slot fires. Zero cells detected at eval across every frame despite GT having 40.

**Root cause identified: `--teacher-force-zpres` creates a massive train/eval mismatch.**

- Training forward: propagate-vs-discover gate uses **GT z_pres[t-1]** — transformer sees "these 40 slots are alive, these 8 dormant". Under that regime the pres_head produces reasonable logits and BCE drives them toward GT.
- Eval forward: gate uses **predicted z_pres = all zeros** — every slot enters discovery. Transformer produces very different queries. Pres_head produces deeply-negative logits.

The model literally never learned to predict z_pres in a self-consistent teacher-free way. Every "IoU" and "silhouette" number in Exp 3-6 was measured on a broken eval forward path. Historical 100k run got IoU 0.127 by pres_head accumulating enough inertia over 100k steps to overcome the `init_bias=-1` PresHead. Our shorter runs never got there.

**Files touched:** none. Diagnostic only.

**Motivation for Exp 7:** drop `--teacher-force-zpres`.

---

### Exp 6: `rebalance_slotcontrast` (2026-07-02 16:02, killed after ckpt step 500)

All Exp 5 changes + **SlotContrast temporal contrastive on z_what** (Manasyan CVPR 2025). InfoNCE at τ=0.1: positive = (`z_what[t, n]`, `z_what[t+1, n]`) same slot across consecutive frames; negatives = other slots at target frame. Only counted for slots alive at both frames. Uses unmatched z_what (slot index = identity by construction under teacher-force z_pres + SAVi bootstrap).

**Config:** All Exp 5 config + `--lambda-slot-contrast 1.0 --slot-contrast-tau 0.1`. `ckpt-every 500` (was 1000) for crash safety.

**New code:** `slot_contrast_loss` in `sim2real/losses/supervised.py`; wired into `PretrainLossConfig` and `pretrain_loss`; CLI flag `--lambda-slot-contrast`.

**Results (in progress):**
| step | eval IoU | silhouette | train contr | train appearG |
|------|----------|------------|-------------|---------------|
| 1    | 0.013    | -0.035     | 3.728       | 0.0149        |
| 100  | —        | —          | 0.921       | 0.0120        |
| 200  | —        | —          | 0.087       | 0.0130        |
| 250  | **0.017** | **-0.022** | —           | —             |

**Comparison at step 250 eval (vs Exp 5):**
- IoU 0.017 vs Exp 5's 0.023 — slight regression (-26%)
- silhouette **-0.022 vs -0.046** — **half as anti-clustered** (this is what SlotContrast targets)
- PSNR/SSIM slightly better

**Interpretation:** SlotContrast is doing what it's meant to — z_what specialization jumped (silhouette went from -0.046 → -0.022 at same step, and train `contr` loss dropped 40× in 200 steps). But comes with a small IoU regression, suggesting the added constraint slightly interferes with localization. Net-positive depends on later evals. Also **~4× slower per step** than Exp 5 (7-10 s/step vs 2.4 s/step) — probably `jnp.diagonal` inside JIT.

---

### Exp 5: `rebalance_symbreak` (2026-07-02 15:05, **crashed** at step 600 / 2000)

Combined recon + glimpse-appearance loss **plus three symmetry-breaking init changes**.

**Config:** many_cells, batch 4, n_max=48, T-curriculum 3→12 over 1000 steps, `--lambda-recon 2.0 --lambda-mask-glimpse 10.0 --lambda-appear-glimpse 10.0 --lambda-where 1.0 --lambda-pres 1.0 --teacher-force-zpres`. Eval every 250.

**Changes vs. prior:**
- `z_where_init`: translation stddev **0.1 → 0.5** (slots start spread across image, not bunched at center). `model.py`
- `SlotTokens.slot_emb`: stddev **0.02 → 0.2** (slot queries start more distinguishable). `slot_transformer.py`
- **SAVi frame-0 GT bootstrap**: during pretrain, always feed `sample.z_where[:, 0]` as the frame-0 residual anchor. New `bootstrap_zwhere0` arg on `SlotVideoModel.__call__`. `model.py`, `train/pretrain.py`

**Results:**
| step | eval IoU | silhouette | PSNR | notes |
|------|----------|------------|------|-------|
| 1    | 0.013    | -0.035     | 20.29 | init only, no training |
| 250  | 0.023    | -0.046     | 21.42 | |
| 500  | 0.024    | -0.061     | 21.41 | plateaued |
| 600  | —        | —          | —    | CUDA illegal-address crash |

Training losses at step 600: loss 0.28, where 0.015, appearG 0.004, mask 0.61.

**Interpretation:** Init changes gave a **big jump-start** — IoU at step 1 is 2.6× the previous best (init only, no training!), and by step 250 we're at IoU 0.023, matching the best step-500 IoU of prior runs. **But the trajectory plateaus after step 250** and silhouette continues to fall (slots increasingly anti-cluster in z_what). Suggests appearG + mask_glimpse alone don't provide the cross-slot decomposition pressure needed to keep improving past the head start.

**Crash:** `CUDA_ERROR_ILLEGAL_ADDRESS` mid-step. Possibly related to `sudo nvidia-smi --lock-gpu-clocks=2500,2900` (which we enabled earlier to fix a 70× throughput regression). Should unlock clocks before next run.

---

### Exp 4: `rebalance_combined` (2026-07-02 14:44, killed at step 250)

Both composite recon **and** glimpse appearance active. Test whether adding recon back on top of appearG rescues decomposition.

**Config:** many_cells, batch 4, T-curriculum, `--lambda-recon 2.0 --lambda-appear-glimpse 10.0 --lambda-mask-glimpse 10.0`. 2000 steps planned, killed at 250.

**Results (step 250 eval):** IoU **0.003**, silhouette -0.107, PSNR 21.37, SSIM 0.135.

**Interpretation:** Adding recon back **helps PSNR/SSIM** (composite fidelity is what pixel MSE is good at) but **does not rescue decomposition** — IoU still ~0.003, silhouette still deeply negative. Nearly identical failure mode to Exp 3. Concluded that **loss weighting is not the actual bottleneck** — the model needs symmetry-breaking at init. Motivated Exp 5.

---

### Exp 3: `rebalance_appear_glimpse` (2026-07-02 14:09, killed at step 500)

Full replacement of composite pixel MSE with per-slot glimpse-space appearance MSE.

**Config:** many_cells, batch 4, T-curriculum 3→12 over 5000, `--lambda-recon 0.0 --lambda-appear-glimpse 10.0 --lambda-mask-glimpse 10.0 --lambda-where 1.0 --lambda-pres 1.0`. 10k steps planned, killed at 500.

**New code:** `glimpse_appear_mse` in `sim2real/losses/supervised.py`. Reads GT image via `stn_read(gt_video, gt_zwhere, gh)`, computes MSE against decoder's `appear_patch`, weighted by GT mask patch (foreground only).

**Results:**
| step | train appearG | eval IoU | silhouette |
|------|---------------|----------|------------|
| 1    | 0.0149        | 0.005    | -0.039     |
| 250  | 0.0084        | —        | —          |
| 500  | 0.0057        | 0.002    | -0.119     |

**Interpretation:** `appearG` descends nicely (48% reduction) but **held-out IoU gets worse** (0.005 → 0.002) and silhouette anti-clusters. Removing composite recon entirely was too aggressive: appearG is a **per-slot** loss with no cross-slot penalty for two slots claiming the same cell. Composite recon (whatever its flaws) provides overlap/decomposition pressure. Killed and moved to Exp 4.

---

### Exp 2: `rebalance_many_cells` (2026-07-02 13:09, killed at step 250 due to CPU contention)

First rebalance attempt: shift loss weights toward recon.

**Config:** `--lambda-recon 20.0 --lambda-mask-glimpse 3.0` (was 2.0 / 10.0). 10k steps planned. Killed at step 250.

**Ran at ~3.5 s/step** — CPU crushed by a separate celegans discrete-solver job (29 cores). Aborted so we could add the eval hook and rerun.

---

### Exp 1: added in-loop held-out eval hook (2026-07-02 13:35)

Structural change to `sim2real/train/pretrain.py`:
- New `PretrainConfig` fields: `eval_every`, `eval_batch_size`, `eval_seed`
- Samples one fixed held-out batch at init with the eval seed
- Every `eval_every` steps: forwards through it at native T (no bootstrap, no teacher forcing), logs `eval/{recon_mse, psnr, ssim, seg_iou, silhouette_zwhat}` to TensorBoard
- CLI: `--eval-every`, `--eval-batch`, `--eval-seed`

Motivation: prior training loops only logged train scalars; test IoU only came from the one-shot `eval_ckpt` at the very end. We needed mid-run signal on the metrics that actually diagnose the failure mode.

---

## Throughput regression (unresolved)

Discovered a **~70× slowdown** vs. historical June runs on identical config:
- Historical: 23 ms/step
- Now: ~1.6 s/step

**What we ruled out:**
- Not external CPU contention (verified with all other jobs killed)
- Not GPU throttling from P-state — during a hot loop, GPU is at P1 with locked clocks 2600/13800 MHz but only **2% utilization**
- Not the Hungarian `pure_callback` (measured at 2.9 ms batched)
- Not the sim (cached jit: 0.3 ms/call)
- Not my new `glimpse_appear_mse` loss (A/B without it still slow)
- Synthetic JAX benchmarks are fast (matmul 30 μs, small graphs 0.2 ms)

**Working hypothesis:** something in `train_step` produces many small kernels with dispatch overhead totaling ~1.6 s. GPU stays idle waiting for the next dispatch. Not yet profiled with `jax.profiler.trace`.

**Env at time of regression:** JAX 0.5.0, jaxlib 0.5.0, flax 0.10.4, optax 0.2.4, numpy 2.1.3. Historical runs (June 2026) may have used a different JAX version — not verified.

**Workaround in use:** enabled `sudo nvidia-smi -pm 1` and `sudo nvidia-smi --lock-gpu-clocks=2500,2900`. This didn't fix throughput (still 1.6 s/step) and possibly caused the Exp 5 crash. **TODO: unlock clocks before next run.**

Deferred; iterating on design at 1.6 s/step for now. `runs/rebalance_symbreak` reached step 500 in ~26 min — tolerable.

---

## Historical baselines (pre-session)

| run                            | date       | steps  | PSNR  | SSIM  | IoU   | silhouette |
|--------------------------------|------------|--------|-------|-------|-------|------------|
| unified_many_cells_v2          | 2026-06-19 | 100000 | 24.16 | 0.520 | 0.127 | -0.034     |
| scheduled/flagella_v2          | 2026-06-19 | 100000 | 19.74 | 0.107 | 0.078 |  0.028     |
| scheduled/multiscale_v2        | 2026-06-19 | 100000 | 23.17 | 0.474 | 0.018 | -0.090     |
| scheduled/worms_v2             | 2026-06-20 | 100000 | 21.55 | 0.141 | 0.129 | -0.015     |

Diagnosis (start of session): all four sims show reconstruction that's plausible on average brightness (PSNR 20-24) but **decomposition is broken** — masks are diffuse blobs, silhouette ~0, IoU an order of magnitude below what should be achievable. Failure mode is textbook Slot-Attention symmetric fixed point + pixel-MSE local minimum, per literature (DINOSAUR, BO-QSA, Invariant Slot Attention, SlotContrast, DSSA).

---

## Design decisions locked in (not yet ablated away)

- **z_style is global per-video**, dim 4. Validated by literature (DSSA 2025, Prompt-Driven DG CVPR 2024, Dittadi ICML 2022). Not per-object.
- **DETR-style per-slot learnable queries** (`SlotTokens`) already present — matches BO-QSA's prescription.
- **Glimpse-space mask supervision** (`glimpse_mask_mse`) is the mask loss we're using (`--lambda-mask-glimpse 10 --lambda-mask 0`). Canvas-space BCE+Dice on masks gets hijacked by the 1:500 foreground:background pixel ratio.
- **Recipe rejected:** pure appearG without composite recon (Exp 3 regressed).
- **Recipe rejected:** loss reweighting alone without addressing init symmetry (Exp 4).

## Open ideas (not tried yet)

- **SlotContrast temporal contrastive** loss between slots (CVPR 2025 Oral). Would directly attack the "slots anti-cluster in z_what" symptom we keep seeing.
- **Warm-start slot queries from Stage-1 GT embeddings** (MetaSlot 2025, our unique advantage). Novel, no published paper.
- **Self-perceptual loss** using our own encoder features as recon target (DINOSAUR-style but bootstrapped on our domain).
- **Mask entropy penalty** to push masks toward binary and avoid diffuse blobs.
- **Multi-scale / Laplacian pyramid MSE** to reduce low-frequency dominance in the composite recon term.
