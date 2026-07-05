# Plan: Next-frame prediction + multi-frame temporal window

## Context

We are changing two things about `SlotVideoModel` (`sim2real/model/model.py`):

1. **Recon objective → next-frame prediction.** Today `L_recon = MSE(composite[t], video[t])`
   — the model reconstructs the same frame it encoded (`losses/losses.py:109`, `:231`). We
   switch to `L_recon = MSE(composite[t], video[t+1])`: the model must *predict* the next
   frame. Applies to **both** regimes (supervised pretrain and unsupervised adapt).

2. **Object inductive bias → temporal permanence via a multi-frame input window.** Each step
   currently attends to a single encoder frame `feats[t]`. We widen the input so the slot
   transition sees a window `[t-k .. t]` with **k=2** (window of 3 frames). This gives the
   transition motion context (velocity/acceleration) so the predicted next pose is grounded in
   observed dynamics rather than a single snapshot.

Slowness of latents (temporal permanence) is **not** a new loss — we lean on the existing
`slot_contrast_loss` (`losses/supervised.py:117`, InfoNCE across consecutive frames, already
wired into `PretrainLossConfig.lambda_slot_contrast`). z_where already changes slowly by
construction (residual `0.5·tanh` cap), and z_style is already one-per-video.

### Decisions confirmed with the user
- **Next-frame recon applies to both pretrain and adapt** (not adapt-only, not a separate term).
- **Window size k=2** (W = 3 frames: `t-2, t-1, t`), edge-padded by repeating frame 0.
- **Read-pose / write-pose split** (the key architectural decision — see §2 below): the glimpse
  appearance is *read* at the object's current pose (frame `t`), the composite is *written* at
  the predicted *next* pose (frame `t+1`). Appearance is permanent, so it is read from the
  frame we actually have; only motion is predicted.
- **Slowness = existing SlotContrast**, no new penalty / no new Prior.kl term.

### Backward compatibility
All changes are behind config flags. Defaults `window_size=1` + `next_frame=False` reproduce
the current model *exactly* (byte-for-byte forward), so existing pretrain runs and the
experiments.md baselines stay reproducible. The new behavior is opt-in.

---

## Approach

### 1. Multi-frame input window (`window_size`, default 1)

**`ModelConfig.window_size: int = 1`.** `W = window_size`; `k = W - 1`.

- **Encoder unchanged.** `feats, pools = vmap(encoder)(video)` still encodes every frame once
  (`model.py:197`). The window is a cheap gather over already-computed features — no re-encode.
- **Window gather in the unrolled loop** (`model.py:366`, the Python `for t in range(T)` loop —
  not `lax.scan`, so direct slicing is fine). For step `t`, build
  `feat_window = feats[max(0,t-k) : t+1]`, left-padded to length `W` by repeating `feats[0]`
  when `t < k`. Pass `feat_window (W, h', w', d)` into `step` instead of `feats[t]`.
- **`SlotTransformer` accepts a window** (`slot_transformer.py`). Change signature to take
  `feat_window (W, h, w, d)`:
  - Add existing `sinusoidal_2d(h, w, d)` spatial PE per frame (as today).
  - Add a **temporal position embedding** distinguishing the W window frames: a learnable table
    `self.param("temporal_pe", ..., (W, d))` (or sinusoidal over W), broadcast added per frame.
  - Flatten to `memory (W·h·w, d)`. Both propagate and discover passes cross-attend this larger
    memory — the *only* structural change to the transformer; the TransformerBlocks are unchanged.
- **Discovery residual mask** (`model.py:236-252`) is computed per spatial cell `(h', w')` from
  the *previous* frame's z_pres/z_where, as today. Tile it across the W window frames →
  `(W·h·w, 1)` and apply `disc_memory = memory * rm_tiled` (`slot_transformer.py:87`). Explained
  regions are suppressed uniformly across the whole window.

Cost note: memory grows W×, so cross-attention cost per step roughly triples at k=2. Given the
existing ~1.6 s/step throughput and the historical T=8 CUDA-illegal-address crashes after
T-curriculum bumps (experiments.md), keep the T-curriculum conservative for the first window run.

### 2. Next-frame prediction with read-pose / write-pose split

**Semantics.** At step `t` the model outputs the object state that renders **frame `t+1`**.
Reconstruction target is `video[t+1]`. The final step `t = T-1` produces a prediction with no
target and is dropped from `L_recon`.

The wrinkle: today one `render_zwhere` does double duty — `stn_read` crops the appearance
glimpse *and* `stn_write` composites it (`model.py:140`, `:164`). For next-frame prediction the
object is at a **new** position in `t+1`, so we split the pose:

- **Read pose = current pose (frame `t`).** `stn_read(video[t], z_where_now, ...)` — appearance
  is permanent and `video[t+1]` is not available at read time. `z_where_now` is exactly what
  `WhereHead` predicts today (residual on the previous frame's pose). Keep `WhereHead` unchanged;
  it is still supervised by `L_where` vs `teacher_zwhere[t]` in pretrain.
- **Write pose = predicted next pose (frame `t+1`).** Add a small **`TransitionHead`** (mirror
  `WhereHead`: `Δ = 0.5·tanh(zero_init_Dense(gelu(Dense(x))))`, zero-init last layer per the v17
  lesson so it starts at identity/no-motion). Input `x = concat([q, slot_h])` so it sees the
  recurrent motion state. `z_where_next = z_where_now + Δ`. Used only for `stn_write`.
- **z_pres, z_what unchanged** — presence and appearance are permanent, so no shift: read z_what
  from the glimpse at `z_where_now`, gate the composite by predicted z_pres as today.

**`_per_slot_head` change** (`model.py:127`): thread two poses — `read_zwhere` for `stn_read`,
`write_zwhere` for the two `stn_write` calls of `appear_canvas` / `mask_appear_canvas`. The
`stop_grad_recon_path` logic applies to `write_zwhere` (render position) and z_pres as today.
`mask_seg_canvas` (the supervised seg target) keeps writing at `read_zwhere` = current pose,
since `L_mask` / `L_mask_glimpse` are supervised against the *current* frame's GT mask.

**Carry** (`model.py:335-340`): carry `z_where_now` as the previous-pose anchor (identical
anchor semantics to today). `z_where_next` is a render-time quantity, not carried — next step
re-derives its own `z_where_now` from `feats[t+1]`. Feed `z_where_next` (or `Δ`) into the GRU
input alongside the existing latents so the recurrent state accumulates velocity.

**Teacher forcing (pretrain).** When `teacher_zwhere` is given: `read_zwhere = teacher[t]`,
`write_zwhere = teacher[t+1]` (GT next pose), so the recon render is GT-consistent. The predicted
`z_where_next` is still produced and trained by an explicit `L_where_next` vs `teacher[t+1]`
(new, optional term — see §3). For `t = T-1`, `teacher[t+1]` is undefined → clamp to `teacher[t]`
and mask that step out of `L_recon` / `L_where_next`.

**Why the split matters (rationale, do not undo):** the rejected single-pose alternative (read
*and* write at the predicted future pose) forces the decoder to hallucinate appearance from a
wrong-location crop of `video[t]` — reintroducing exactly the mean-image / decode-from-nothing
degeneracy that the glimpse-space losses (Exp 3-8) were built to avoid.

### 3. Loss changes (`losses/losses.py`)

- **`PretrainLossConfig.next_frame: bool = False`** and **`AdaptLossConfig.next_frame: bool = False`.**
- **Recon target shift.** When `next_frame`:
  `L_recon = recon_mse(out.composite[:-1], sample.video[1:])` (drop last predicted frame). Add a
  small helper or inline; `recon_mse` itself is unchanged (`losses/recon.py`).
- **New optional `lambda_where_next` + `L_where_next`** (pretrain only): matched MSE of the
  predicted `z_where_next` against `sample.z_where` shifted by one frame (`sample.z_where[1:]`),
  masked by `sample.z_pres[1:]`. Reuses the existing Hungarian `perm` (matching is on
  `z_where_now`, identity-stable). Directly trains the `TransitionHead`. Default weight 0 so it
  is opt-in / ablatable; expose `z_where_next` via `ModelOut.aux["z_where_next"]`.
- **SlotContrast** already handles slowness; nothing to add for pretrain. **Optionally** add the
  same 3 lines to `adapt_loss` (`losses.py:224`) using unmatched `out.z_what` if we want slowness
  pressure during adapt too — flagged as a follow-up, not part of the core change.

### 4. Trainer + CLI plumbing

- **`sim2real/train/pretrain.py`**: pass `window_size` into `ModelConfig`; pass `next_frame`
  (+ `lambda_where_next`) into `PretrainLossConfig`. Handle the teacher-force `t+1` shift and the
  last-frame masking when building teacher slices.
- **`sim2real/train/adapt.py`**: pass `window_size` into `ModelConfig`, `next_frame` into
  `AdaptLossConfig`.
- **`sim2real/scripts/pretrain.py`**: add `--window-size N` (default 1), `--next-frame` flag,
  `--lambda-where-next F` (default 0).
- Mirror the flags on the adapt CLI if one exists.

### 5. `ModelOut` / types

Add `aux["z_where_next"]` (`(T, N, 5)`). Everything else unchanged. `out.z_where` continues to
mean the *current-frame* pose (`z_where_now`) so all existing supervised losses, matching, and
eval metrics (IoU on `z_where`) keep their current semantics.

---

## Build order (TaskList)

1. **`ModelConfig.window_size` + window gather + `SlotTransformer` temporal window.** Encode
   once, gather `[t-k..t]` in the loop, temporal PE, W× memory in both passes, tile residual
   mask. Verify `window_size=1` reproduces current forward bit-for-bit.
2. **`TransitionHead` + read/write-pose split in `_per_slot_head`.** New head, two poses threaded
   through the vmap, `z_where_next` in carry/GRU input and `ModelOut.aux`. Verify
   `next_frame=False` path unchanged (write_zwhere == read_zwhere).
3. **Next-frame recon in both loss aggregators.** `next_frame` flag, `composite[:-1]` vs
   `video[1:]`, plus optional `L_where_next`.
4. **Trainer + CLI plumbing** (pretrain + adapt): teacher `t+1` shift, last-frame masking, flags.
5. **Smoke test + first run.** `pytest tests/`; overfit-one-video sanity that next-frame recon
   descends; then a short many_cells run at `--window-size 3 --next-frame --lambda-slot-contrast 1.0`
   with a conservative T-curriculum. Log to experiments.md.

## Risks / watch-items
- **Graph size / CUDA crash.** W× memory + longer unrolled T is more XLA to compile; the
  historical `CUDA_ERROR_ILLEGAL_ADDRESS` fired after T-curriculum bumps into T=8. Keep T ≤ 6-7
  for the first window run.
- **Throughput.** Already ~1.6 s/step; window ~triples cross-attention. Accept for design
  iteration; profile only if it blocks.
- **Read-pose validity under adapt.** Appearance read at `z_where_now` is only correct if
  `z_where_now` localizes the object in `video[t]`. Pretrain pins it via `L_where`; adapt relies
  on that transferring. This *is* the sim2real hypothesis — recon-at-`t+1` then trains the
  transition on the new prior. If `z_where_now` drifts during adapt, appearance reads go wrong;
  watch eval IoU on the adapt set.
- **Note for future Claude:** CLAUDE.md still points the plan at a stale path
  (`i-m-starting-a-new-purring-lemon.md`, which does not exist). This doc + `experiments.md` are
  the live design records.
