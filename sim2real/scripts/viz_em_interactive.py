"""Generate an interactive HTML visualization of NEM iterations.

Runs the model manually on one video, recording per-iteration:
  - image (fixed background)
  - feature grid dimensions
  - per-slot responsibility over grid cells (softmax-over-slots weights)
  - per-slot z_where (pos, scale), z_pres

Writes a self-contained HTML with embedded JSON + vanilla JS + canvas.
Controls: iteration slider, slot selector, single-slot vs all-slots toggle.

Usage:
    PYTHONPATH=. python3 -m sim2real.scripts.viz_em_interactive \
        --ckpt runs/nem_stride2_10k/ckpts/step_10000.pkl \
        --sim many_cells_fast --n-max 18 \
        --glimpse-size 16 --d-model 128 --n-transformer-layers 3 --stem-strides 2 1 1 \
        --use-neural-em --anchor-init-fixed --z-what-init-std 1.0 --nem-attn-temp 0.25 \
        --n-iters-diagnostic 10 --seed 1
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

from sim2real.model.encoder import FrameEncoder
from sim2real.model.model import ModelConfig, SlotVideoModel
from sim2real.model.neural_em import NeuralEMRefiner, _normalized_grid, _slot_pos_embed
from sim2real.model.posenc import sinusoidal_2d
from sim2real.sim.api import build_sim
from sim2real.train.ckpt import load as ckpt_load
from sim2real.types import SimSample


def slice_to_model(batch, Nm):
    return SimSample(
        video=batch.video, z_where=batch.z_where[:, :, :Nm],
        z_pres=batch.z_pres[:, :, :Nm], z_style=batch.z_style,
        masks=batch.masks[:, :, :Nm],
        z_what=None if batch.z_what is None else batch.z_what[:, :Nm],
        meta=batch.meta,
    )


def one_em_step_with_resp(K, V, pixel_pos, z_where, z_pres, z_what, ne_params, cfg,
                          attn_temp: float, use_bg_slot: bool = False):
    """Run one NEM E+M step and return responsibility along with new latents.

    Duplicates NeuralEMRefiner.__call__ but manually applies the refiner params and
    returns the intermediate responsibility matrix.
    """
    d_model = cfg.d_model
    d_pos = 32
    z_where_delta_scale = 0.05

    # --- E-step (Q from z_where + z_what, softmax over slots) ---
    pos_xy = jnp.stack([jnp.tanh(z_where[:, 3]), jnp.tanh(z_where[:, 4])], axis=-1)
    pos_emb = _slot_pos_embed(pos_xy, d_pos)
    state_repr = jnp.concatenate([pos_emb, z_what], axis=-1)
    ln_p = ne_params["refiner"]["state_norm"]
    mu = state_repr.mean(-1, keepdims=True)
    var = state_repr.var(-1, keepdims=True)
    state_repr_n = (state_repr - mu) / jnp.sqrt(var + 1e-6) * ln_p["scale"] + ln_p["bias"]
    Q = state_repr_n @ ne_params["refiner"]["q_proj"]["kernel"]                            # (N, d_model)

    scale = d_model ** -0.5
    logits = (Q @ K.T) * scale / attn_temp + jnp.log(z_pres + 1e-6)[:, None]               # (N, L)
    if use_bg_slot:
        bg_logit = ne_params["refiner"]["bg_logit"]
        bg_row = jnp.broadcast_to(bg_logit, (1, logits.shape[-1]))
        logits_full = jnp.concatenate([logits, bg_row], axis=0)                            # (N+1, L)
        resp_full = jax.nn.softmax(logits_full, axis=0)
        resp = resp_full[:-1]                                                              # (N, L)
    else:
        resp = jax.nn.softmax(logits, axis=0)                                              # (N, L)

    # --- M-step ---
    mass = resp.sum(axis=-1)
    weighted_pos = resp @ pixel_pos
    centroid = weighted_pos / (mass[:, None] + 1e-6)
    diff = pixel_pos[None] - centroid[:, None]
    weighted_var = (resp[:, :, None] * (diff ** 2)).sum(axis=1) / (mass[:, None] + 1e-6)
    std = jnp.sqrt(weighted_var + 1e-6)

    slot_feat = (resp @ V) / (mass[:, None] + 1e-6)
    ln_sf = ne_params["refiner"]["slot_feat_norm"]
    mu = slot_feat.mean(-1, keepdims=True)
    var = slot_feat.var(-1, keepdims=True)
    slot_feat_norm = (slot_feat - mu) / jnp.sqrt(var + 1e-6) * ln_sf["scale"] + ln_sf["bias"]

    dh = ne_params["refiner"]["delta_hidden"]
    delta_hidden = jax.nn.gelu(slot_feat_norm @ dh["kernel"] + dh["bias"])
    delta_out = ne_params["refiner"]["delta_out"]
    delta = delta_hidden @ delta_out["kernel"] + delta_out["bias"]

    theta_raw = z_where[:, 2] + z_where_delta_scale * jnp.tanh(delta[:, 2])
    _S_CLIP = (0.02, 0.95)
    _T_CLIP = 0.98
    tx_raw = jnp.arctanh(jnp.clip(centroid[:, 0], -_T_CLIP, _T_CLIP))
    ty_raw = jnp.arctanh(jnp.clip(centroid[:, 1], -_T_CLIP, _T_CLIP))
    sx_raw = jax.scipy.special.logit(jnp.clip(std[:, 0], _S_CLIP[0], _S_CLIP[1]))
    sy_raw = jax.scipy.special.logit(jnp.clip(std[:, 1], _S_CLIP[0], _S_CLIP[1]))
    base_zwhere = jnp.stack([sx_raw, sy_raw, theta_raw, tx_raw, ty_raw], axis=-1)
    delta_xy = jnp.concatenate([delta[:, 0:2], jnp.zeros_like(delta[:, 2:3]), delta[:, 3:5]], axis=-1)
    z_where_new = base_zwhere + z_where_delta_scale * jnp.tanh(delta_xy)

    Zw = z_what.shape[-1]
    what_pre_p = ne_params["refiner"]["what_pre"]
    what_pre = slot_feat_norm @ what_pre_p["kernel"] + what_pre_p["bias"]
    gru_p = ne_params["refiner"]["what_gru"]
    # GRU manual: hidden state = z_what (current-iter), input = what_pre
    hn = gru_p["hn"]
    ir = gru_p["ir"]; iz = gru_p["iz"]; in_ = gru_p["in"]
    hr = gru_p["hr"]; hz = gru_p["hz"]
    r = jax.nn.sigmoid(what_pre @ ir["kernel"] + ir["bias"] + z_what @ hr["kernel"])
    z = jax.nn.sigmoid(what_pre @ iz["kernel"] + iz["bias"] + z_what @ hz["kernel"])
    n = jax.nn.tanh(what_pre @ in_["kernel"] + in_["bias"] + r * (z_what @ hn["kernel"] + hn["bias"]))
    z_what_new = (1.0 - z) * n + z * z_what
    # Residual MLP
    ln_pm = ne_params["refiner"]["post_mlp_norm"]
    mu = z_what_new.mean(-1, keepdims=True)
    var = z_what_new.var(-1, keepdims=True)
    znorm = (z_what_new - mu) / jnp.sqrt(var + 1e-6) * ln_pm["scale"] + ln_pm["bias"]
    pmi = ne_params["refiner"]["post_mlp_in"]
    pmo = ne_params["refiner"]["post_mlp_out"]
    z_what_new = z_what_new + (jax.nn.gelu(znorm @ pmi["kernel"] + pmi["bias"])
                               @ pmo["kernel"] + pmo["bias"])

    thresh = ne_params["refiner"]["mass_thresh"]
    temp = ne_params["refiner"]["mass_temp"]
    z_pres_new = jax.nn.sigmoid((mass - thresh) / (jnp.abs(temp) + 0.1))

    return z_where_new, z_pres_new, z_what_new, resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--n-max", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-iters-diagnostic", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-transformer-layers", type=int, default=3)
    ap.add_argument("--glimpse-size", type=int, default=16)
    ap.add_argument("--stem-strides", type=int, nargs="+", default=[2, 2, 1])
    ap.add_argument("--use-neural-em", action="store_true")
    ap.add_argument("--anchor-init-fixed", action="store_true")
    ap.add_argument("--z-what-init-std", type=float, default=0.2)
    ap.add_argument("--nem-attn-temp", type=float, default=1.0)
    ap.add_argument("--nem-use-bg-slot", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert args.use_neural_em, "only NEM is supported"

    ck = ckpt_load(args.ckpt)
    params = ck["params"]
    cfg = ModelConfig(
        n_max=args.n_max, d_model=args.d_model, n_heads=4,
        n_transformer_layers=args.n_transformer_layers,
        z_what_dim=64, z_style_dim=4, glimpse_size=args.glimpse_size,
        stem_channels=(16, 32, 64), stem_strides=tuple(args.stem_strides),
        n_groups=1, use_background=True, bg_base_res=4, bg_channels=(8,),
        use_neural_em=True,
        anchor_init_fixed=args.anchor_init_fixed,
        z_what_init_std=args.z_what_init_std,
        nem_attn_temp=args.nem_attn_temp,
        nem_use_bg_slot=args.nem_use_bg_slot,
    )
    model = SlotVideoModel(cfg=cfg)
    batch_fn, _ = build_sim(args.sim)
    key = jax.random.key(args.seed)
    batch = batch_fn(key, 1)
    bm = slice_to_model(batch, args.n_max)
    video = bm.video[0]
    H, W = video.shape[1], video.shape[2]
    gt_pos0 = np.tanh(np.asarray(bm.z_where)[0, 0, :, 3:5])
    gt_alive0 = np.asarray(bm.z_pres)[0, 0]

    # Encode frame 0.
    encoder = FrameEncoder(d_model=cfg.d_model, n_vit_layers=cfg.n_vit_layers,
                           stem_channels=tuple(cfg.stem_channels),
                           stem_strides=tuple(cfg.stem_strides))
    enc_params = {"params": params["params"]["encoder"]}
    feat_grid, _ = encoder.apply(enc_params, video[0])
    h_feat, w_feat = feat_grid.shape[0], feat_grid.shape[1]

    # Init slot state.
    if cfg.anchor_init_fixed:
        side = int(math.ceil(math.sqrt(cfg.n_max)))
        lin = jnp.linspace(-0.7, 0.7, side)
        gy, gx = jnp.meshgrid(lin, lin, indexing="ij")
        pts = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)[:cfg.n_max]
        pos_raw = jnp.arctanh(jnp.clip(pts, -0.98, 0.98))
        scale_raw = jnp.full((cfg.n_max, 2), -2.2)
        theta_raw = jnp.zeros((cfg.n_max, 1))
        z_where_init = jnp.concatenate([scale_raw, theta_raw, pos_raw], axis=-1)
    else:
        z_where_init = params["params"]["z_where_init"]
    z_what_init = params["params"]["z_what_init"]
    z_pres_init = jnp.full((cfg.n_max,), 0.5)

    ne_params = params["params"]["neural_em"]

    # Precompute K, V.
    pe = sinusoidal_2d(h_feat, w_feat, feat_grid.shape[2])
    feats_flat = (feat_grid + pe).reshape(-1, feat_grid.shape[2])
    ln_p = ne_params["feat_norm"]
    mu = feats_flat.mean(-1, keepdims=True)
    var = feats_flat.var(-1, keepdims=True)
    feats_flat_n = (feats_flat - mu) / jnp.sqrt(var + 1e-6) * ln_p["scale"] + ln_p["bias"]
    K = feats_flat_n @ ne_params["k_proj"]["kernel"]
    V = feats_flat_n @ ne_params["v_proj"]["kernel"]
    pixel_pos = _normalized_grid(h_feat, w_feat)

    # Iterate.
    z_where = z_where_init
    z_pres = z_pres_init
    z_what = z_what_init
    traj_zwhere = [np.asarray(z_where)]
    traj_zpres = [np.asarray(z_pres)]
    traj_resp = []   # length I (one less than states)
    for _ in range(args.n_iters_diagnostic):
        z_where, z_pres, z_what, resp = one_em_step_with_resp(
            K, V, pixel_pos, z_where, z_pres, z_what, ne_params, cfg, args.nem_attn_temp,
            use_bg_slot=cfg.nem_use_bg_slot,
        )
        traj_zwhere.append(np.asarray(z_where))
        traj_zpres.append(np.asarray(z_pres))
        traj_resp.append(np.asarray(resp).reshape(cfg.n_max, h_feat, w_feat))

    zw_traj = np.stack(traj_zwhere)                                                        # (I+1, N, 5)
    zp_traj = np.stack(traj_zpres)                                                         # (I+1, N)
    resp_traj = np.stack(traj_resp)                                                        # (I, N, h, w)

    # Per-iter composite: run the model with n_transformer_layers = i for i=1..I. Params
    # are shared across NEM iterations (single refiner reused), so the same params work
    # regardless of n_iters at inference.
    print(f"computing per-iter composite ({args.n_iters_diagnostic} forward passes)...", flush=True)
    composite_traj = []                                                                    # list of (H, W) grayscale
    for i in range(1, args.n_iters_diagnostic + 1):
        cfg_i = ModelConfig(
            n_max=cfg.n_max, d_model=cfg.d_model, n_heads=4,
            n_transformer_layers=i,
            z_what_dim=64, z_style_dim=4, glimpse_size=args.glimpse_size,
            stem_channels=(16, 32, 64), stem_strides=tuple(args.stem_strides),
            n_groups=1, use_background=True, bg_base_res=4, bg_channels=(8,),
            use_neural_em=True,
            anchor_init_fixed=cfg.anchor_init_fixed,
            z_what_init_std=cfg.z_what_init_std,
            nem_attn_temp=cfg.nem_attn_temp,
            nem_use_bg_slot=cfg.nem_use_bg_slot,
        )
        model_i = SlotVideoModel(cfg=cfg_i)
        out_i = model_i.apply(params, video, key)
        composite_traj.append(np.asarray(out_i.composite[0, ..., 0]))                     # frame 0
    composite_arr = np.stack(composite_traj)                                               # (I, H, W)

    # Encode composites as base64 PNGs.
    import matplotlib.pyplot as plt
    composite_b64_list = []
    for i in range(args.n_iters_diagnostic):
        buf = io.BytesIO()
        plt.imsave(buf, np.clip(composite_arr[i], 0, 1), cmap="gray", vmin=0, vmax=1, format="png")
        composite_b64_list.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    # Serialize.
    # Encode image as base64 PNG so HTML can display it inline.
    import matplotlib.pyplot as plt
    fig_img_arr = np.asarray(video[0, ..., 0])
    buf = io.BytesIO()
    plt.imsave(buf, fig_img_arr, cmap="gray", vmin=0, vmax=1, format="png")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
        "img_b64": img_b64,
        "composite_b64": composite_b64_list,   # length I, per-iter reconstruction PNG
        "img_H": int(H), "img_W": int(W),
        "feat_h": int(h_feat), "feat_w": int(w_feat),
        "n_slots": int(cfg.n_max),
        "n_iters": int(args.n_iters_diagnostic),
        "trained_iters": int(cfg.n_transformer_layers),
        "attn_temp": float(args.nem_attn_temp),
        "match_radius": 0.16,   # for coverage classification (~5 px in 64x64)
        # (I+1, N, 2) — positions in normalized [-1,1] (after tanh)
        "pos": np.tanh(zw_traj[:, :, 3:5]).tolist(),
        # (I+1, N, 2) — scale after sigmoid
        "scale": (1.0 / (1.0 + np.exp(-zw_traj[:, :, 0:2]))).tolist(),
        # (I+1, N) — z_pres
        "z_pres": zp_traj.tolist(),
        # (I, N, h, w) — responsibility per slot (initial iter has no resp)
        "resp": resp_traj.tolist(),
        # GT reference
        "gt_pos": gt_pos0.tolist(),
        "gt_alive": gt_alive0.tolist(),
        "ckpt": args.ckpt,
    }
    payload_json = json.dumps(payload)

    html = _make_html(payload_json)

    out_path = args.out or os.path.splitext(args.ckpt)[0] + f"_em_interactive_seed{args.seed}.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path}")
    print(f"  data: {len(payload_json)/1024:.1f} KB (I={args.n_iters_diagnostic}, N={cfg.n_max}, "
          f"grid={h_feat}x{w_feat})")


def _make_html(payload_json: str) -> str:
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NEM E-M iteration visualizer</title>
<style>
  body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
  .row { display: flex; gap: 20px; align-items: flex-start; }
  .panel { background: white; padding: 15px; border-radius: 6px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  canvas { border: 1px solid #ccc; image-rendering: pixelated; }
  .controls { min-width: 300px; }
  label { display: block; margin-top: 12px; font-weight: bold; }
  .info { font-family: monospace; font-size: 12px; margin-top: 10px; color: #666; }
  input[type=range] { width: 100%; }
  select, button { padding: 6px; font-size: 14px; }
  .slot-btn { display: inline-block; margin: 2px; padding: 4px 8px; cursor: pointer;
              border: 1px solid #aaa; border-radius: 4px; font-family: monospace;
              font-size: 12px; user-select: none; }
  .slot-btn.selected { background: #333; color: white; }
  .slot-btn.dead { opacity: 0.4; }
  h1 { font-size: 18px; margin: 0 0 10px 0; }
  h2 { font-size: 14px; margin: 0 0 6px 0; color: #666; }
</style>
</head>
<body>
<div class="row">
  <div>
    <div class="row" style="margin-bottom: 15px;">
      <div class="panel">
        <h1>Image + slot positions</h1>
        <canvas id="canvas_img" width="384" height="384"></canvas>
        <div class="info" id="info_img"></div>
      </div>
      <div class="panel">
        <h1>Responsibility (selected slot)</h1>
        <canvas id="canvas_resp" width="384" height="384"></canvas>
        <div class="info" id="info_resp"></div>
      </div>
    </div>
    <div class="row">
      <div class="panel">
        <h1>Cell coverage</h1>
        <canvas id="canvas_cov" width="384" height="384"></canvas>
        <div class="info" id="info_cov"></div>
      </div>
      <div class="panel">
        <h1>Reconstruction</h1>
        <canvas id="canvas_recon" width="384" height="384"></canvas>
        <div class="info" id="info_recon"></div>
      </div>
    </div>
  </div>
  <div class="panel controls">
    <h1>Controls</h1>
    <label>Iteration: <span id="iter_label">0</span></label>
    <input id="iter_slider" type="range" min="0" max="10" value="0">
    <button id="btn_play">▶ Play</button>
    <label>Selected slot:</label>
    <div id="slot_buttons"></div>
    <label>Display:</label>
    <div>
      <input type="checkbox" id="chk_all_slots" checked>
      <label style="display:inline; font-weight:normal;" for="chk_all_slots">All slots</label>
    </div>
    <div>
      <input type="checkbox" id="chk_gt" checked>
      <label style="display:inline; font-weight:normal;" for="chk_gt">GT cell markers (red X)</label>
    </div>
    <div>
      <input type="checkbox" id="chk_grid" checked>
      <label style="display:inline; font-weight:normal;" for="chk_grid">Feature grid overlay</label>
    </div>
    <div>
      <input type="checkbox" id="chk_scale">
      <label style="display:inline; font-weight:normal;" for="chk_scale">Show slot scale ellipses</label>
    </div>
    <div>
      <input type="checkbox" id="chk_hide_dead">
      <label style="display:inline; font-weight:normal;" for="chk_hide_dead">Hide slots with z_pres &lt; 0.5</label>
    </div>
    <div class="info" id="info_state"></div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const HUE_STEP = 360 / DATA.n_slots;
const DISPLAY_SIZE = 384;

function slotColor(k, alpha) {
  return "hsla(" + ((k * HUE_STEP) % 360) + ", 80%, 50%, " + (alpha == null ? 1 : alpha) + ")";
}

let state = { iter: 0, selected_slot: 0, playing: false, playTimer: null };

// Load base image + all per-iter composites.
const img = new Image();
const composites = [];
let loadedCount = 0;
const totalToLoad = 1 + DATA.n_iters;
function onLoad() { loadedCount++; if (loadedCount === totalToLoad) render(); }
img.onload = onLoad;
img.src = "data:image/png;base64," + DATA.img_b64;
for (let i = 0; i < DATA.n_iters; i++) {
  const c = new Image();
  c.onload = onLoad;
  c.src = "data:image/png;base64," + DATA.composite_b64[i];
  composites.push(c);
}

// Build slot buttons.
const slotButtons = document.getElementById("slot_buttons");
for (let k = 0; k < DATA.n_slots; k++) {
  const b = document.createElement("span");
  b.className = "slot-btn";
  b.textContent = k;
  b.style.color = slotColor(k);
  b.onclick = () => { state.selected_slot = k; renderSlotButtons(); render(); };
  b.dataset.slot = k;
  slotButtons.appendChild(b);
}
function renderSlotButtons() {
  for (const b of slotButtons.children) {
    const k = parseInt(b.dataset.slot);
    b.classList.toggle("selected", k === state.selected_slot);
    // Mark permanently-dead slots (all iters have z_pres < 0.5)
    let everAlive = false;
    for (let i = 0; i < DATA.z_pres.length; i++) {
      if (DATA.z_pres[i][k] > 0.5) { everAlive = true; break; }
    }
    b.classList.toggle("dead", !everAlive);
  }
}
renderSlotButtons();

// Iter slider.
const slider = document.getElementById("iter_slider");
slider.max = DATA.n_iters;
slider.oninput = e => { state.iter = parseInt(e.target.value); render(); };

// Play button.
const playBtn = document.getElementById("btn_play");
playBtn.onclick = () => {
  if (state.playing) {
    clearInterval(state.playTimer);
    state.playing = false;
    playBtn.textContent = "▶ Play";
  } else {
    state.playing = true;
    playBtn.textContent = "⏸ Pause";
    state.playTimer = setInterval(() => {
      state.iter = (state.iter + 1) % (DATA.n_iters + 1);
      slider.value = state.iter;
      render();
    }, 500);
  }
};

// Checkboxes trigger re-render.
for (const id of ["chk_all_slots", "chk_gt", "chk_grid", "chk_scale", "chk_hide_dead"]) {
  document.getElementById(id).onchange = () => render();
}

function posToPx(p) {
  // p in [-1, 1] -> [0, DISPLAY_SIZE-1]
  return [ (p[0] + 1) / 2 * (DISPLAY_SIZE - 1),
           (p[1] + 1) / 2 * (DISPLAY_SIZE - 1) ];
}

function render() {
  document.getElementById("iter_label").textContent = state.iter +
    (state.iter === DATA.trained_iters ? "  ★ trained" : "");

  // ---- Canvas 1: image + slot overlay ----
  const c1 = document.getElementById("canvas_img");
  const ctx1 = c1.getContext("2d");
  ctx1.imageSmoothingEnabled = false;
  ctx1.drawImage(img, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);

  // Feature grid overlay.
  if (document.getElementById("chk_grid").checked) {
    ctx1.strokeStyle = "rgba(0,0,255,0.15)";
    ctx1.lineWidth = 1;
    const step = DISPLAY_SIZE / DATA.feat_w;
    for (let i = 0; i <= DATA.feat_w; i++) {
      ctx1.beginPath();
      ctx1.moveTo(i * step, 0); ctx1.lineTo(i * step, DISPLAY_SIZE);
      ctx1.moveTo(0, i * step); ctx1.lineTo(DISPLAY_SIZE, i * step);
      ctx1.stroke();
    }
  }

  // GT markers.
  if (document.getElementById("chk_gt").checked) {
    ctx1.strokeStyle = "red"; ctx1.lineWidth = 2;
    for (let g = 0; g < DATA.gt_pos.length; g++) {
      if (DATA.gt_alive[g] < 0.5) continue;
      const [x, y] = posToPx(DATA.gt_pos[g]);
      ctx1.beginPath();
      ctx1.moveTo(x - 5, y - 5); ctx1.lineTo(x + 5, y + 5);
      ctx1.moveTo(x + 5, y - 5); ctx1.lineTo(x - 5, y + 5);
      ctx1.stroke();
    }
  }

  // Slot positions.
  const showAll = document.getElementById("chk_all_slots").checked;
  const showScale = document.getElementById("chk_scale").checked;
  const hideDead = document.getElementById("chk_hide_dead").checked;
  const positions = DATA.pos[state.iter];
  const scales = DATA.scale[state.iter];
  const zps = DATA.z_pres[state.iter];
  for (let k = 0; k < DATA.n_slots; k++) {
    if (!showAll && k !== state.selected_slot) continue;
    const alive = zps[k] > 0.5;
    if (hideDead && !alive) continue;
    const [x, y] = posToPx(positions[k]);
    const highlight = k === state.selected_slot;
    // Scale ellipse.
    if (showScale && alive) {
      const rx = scales[k][0] * DISPLAY_SIZE / 2;
      const ry = scales[k][1] * DISPLAY_SIZE / 2;
      ctx1.strokeStyle = slotColor(k, 0.5);
      ctx1.lineWidth = 1;
      ctx1.beginPath();
      ctx1.ellipse(x, y, rx, ry, 0, 0, 2 * Math.PI);
      ctx1.stroke();
    }
    // Dot.
    ctx1.fillStyle = slotColor(k, alive ? 0.9 : 0.3);
    ctx1.strokeStyle = highlight ? "yellow" : "black";
    ctx1.lineWidth = highlight ? 3 : 1;
    ctx1.beginPath();
    ctx1.arc(x, y, alive ? 10 : 6, 0, 2 * Math.PI);
    ctx1.fill(); ctx1.stroke();
    // Label.
    ctx1.fillStyle = "white";
    ctx1.font = "bold 11px sans-serif";
    ctx1.textAlign = "center"; ctx1.textBaseline = "middle";
    ctx1.fillText(k, x, y);
  }

  document.getElementById("info_img").textContent =
    "iter " + state.iter + " / " + DATA.n_iters + " | " +
    "temp=" + DATA.attn_temp.toFixed(2) + " | " +
    "image " + DATA.img_H + "x" + DATA.img_W + " | " +
    "feat grid " + DATA.feat_h + "x" + DATA.feat_w;

  // ---- Canvas 2: responsibility heatmap ----
  const c2 = document.getElementById("canvas_resp");
  const ctx2 = c2.getContext("2d");
  ctx2.imageSmoothingEnabled = false;
  ctx2.drawImage(img, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
  ctx2.globalCompositeOperation = "source-over";
  ctx2.fillStyle = "rgba(255,255,255,0.4)";
  ctx2.fillRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);

  if (state.iter > 0) {
    const respIter = state.iter - 1;   // resp is length I, indexed 0..I-1
    const respMap = DATA.resp[respIter][state.selected_slot];   // (feat_h, feat_w)
    // Find max for normalization.
    let maxVal = 0;
    for (let i = 0; i < DATA.feat_h; i++) for (let j = 0; j < DATA.feat_w; j++)
      if (respMap[i][j] > maxVal) maxVal = respMap[i][j];

    const stepH = DISPLAY_SIZE / DATA.feat_h;
    const stepW = DISPLAY_SIZE / DATA.feat_w;
    // Draw responsibility as red overlay.
    for (let i = 0; i < DATA.feat_h; i++) {
      for (let j = 0; j < DATA.feat_w; j++) {
        const v = respMap[i][j] / (maxVal + 1e-6);
        ctx2.fillStyle = "rgba(255,0,0," + (v * 0.7) + ")";
        ctx2.fillRect(j * stepW, i * stepH, stepW, stepH);
      }
    }
    // Slot position marker.
    const [x, y] = posToPx(DATA.pos[state.iter][state.selected_slot]);
    ctx2.fillStyle = slotColor(state.selected_slot);
    ctx2.strokeStyle = "black"; ctx2.lineWidth = 2;
    ctx2.beginPath(); ctx2.arc(x, y, 8, 0, 2 * Math.PI);
    ctx2.fill(); ctx2.stroke();

    let entropy = 0;
    const respFlat = [];
    for (let i = 0; i < DATA.feat_h; i++) for (let j = 0; j < DATA.feat_w; j++)
      respFlat.push(respMap[i][j]);
    const sumV = respFlat.reduce((a, b) => a + b, 0);
    if (sumV > 0) {
      for (const v of respFlat) if (v > 0) entropy -= (v / sumV) * Math.log(v / sumV);
    }

    document.getElementById("info_resp").textContent =
      "slot " + state.selected_slot + " | z_pres=" + zps[state.selected_slot].toFixed(3) +
      " | mass=" + sumV.toFixed(2) +
      " | entropy=" + entropy.toFixed(2) +
      " | scale=(" + scales[state.selected_slot][0].toFixed(3) +
      "," + scales[state.selected_slot][1].toFixed(3) + ")";
  } else {
    ctx2.fillStyle = "rgba(0,0,0,0.5)"; ctx2.font = "18px sans-serif";
    ctx2.textAlign = "center";
    ctx2.fillText("Iteration 0 (initial state — no responsibility computed yet)",
                  DISPLAY_SIZE / 2, DISPLAY_SIZE / 2);
    document.getElementById("info_resp").textContent =
      "slot " + state.selected_slot + " (initial state)";
  }

  // ---- Canvas 3: cell coverage ----
  const c3 = document.getElementById("canvas_cov");
  const ctx3 = c3.getContext("2d");
  ctx3.imageSmoothingEnabled = false;
  ctx3.drawImage(img, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
  // Dim underneath so overlays stand out.
  ctx3.fillStyle = "rgba(255,255,255,0.25)";
  ctx3.fillRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);

  // Collect alive pred slots at this iter.
  const aliveSlots = [];
  for (let k = 0; k < DATA.n_slots; k++) {
    if (zps[k] > 0.5) aliveSlots.push({ id: k, pos: positions[k] });
  }
  // Collect alive GT cells.
  const aliveGT = [];
  for (let g = 0; g < DATA.gt_pos.length; g++) {
    if (DATA.gt_alive[g] > 0.5) aliveGT.push({ id: g, pos: DATA.gt_pos[g] });
  }

  // For each GT: find alive slots within match_radius.
  const R = DATA.match_radius;
  let missed = 0, dup = 0, phantom = 0;
  const gtCovered = aliveGT.map(g => {
    const near = aliveSlots.filter(s => {
      const dx = s.pos[0] - g.pos[0], dy = s.pos[1] - g.pos[1];
      return Math.sqrt(dx*dx + dy*dy) < R;
    });
    return near;
  });
  for (const cov of gtCovered) { if (cov.length === 0) missed++; if (cov.length >= 2) dup++; }
  for (const s of aliveSlots) {
    const near = aliveGT.some(g => {
      const dx = s.pos[0] - g.pos[0], dy = s.pos[1] - g.pos[1];
      return Math.sqrt(dx*dx + dy*dy) < R;
    });
    if (!near) phantom++;
  }

  // Draw GT (red circles) — missed cells get orange outer ring.
  for (let gi = 0; gi < aliveGT.length; gi++) {
    const g = aliveGT[gi];
    const [x, y] = posToPx(g.pos);
    if (gtCovered[gi].length === 0) {
      ctx3.strokeStyle = "orange"; ctx3.lineWidth = 3;
      ctx3.beginPath(); ctx3.arc(x, y, 14, 0, 2 * Math.PI); ctx3.stroke();
    }
    ctx3.strokeStyle = "red"; ctx3.lineWidth = 2;
    ctx3.beginPath(); ctx3.arc(x, y, 10, 0, 2 * Math.PI); ctx3.stroke();
  }
  // Draw alive pred slots (blue dots with slot id).
  for (const s of aliveSlots) {
    const [x, y] = posToPx(s.pos);
    const near = aliveGT.some(g => {
      const dx = s.pos[0] - g.pos[0], dy = s.pos[1] - g.pos[1];
      return Math.sqrt(dx*dx + dy*dy) < R;
    });
    ctx3.fillStyle = near ? "dodgerblue" : "magenta";  // phantom = magenta
    ctx3.strokeStyle = "black"; ctx3.lineWidth = 1;
    ctx3.beginPath(); ctx3.arc(x, y, 8, 0, 2 * Math.PI); ctx3.fill(); ctx3.stroke();
    ctx3.fillStyle = "white";
    ctx3.font = "bold 10px sans-serif";
    ctx3.textAlign = "center"; ctx3.textBaseline = "middle";
    ctx3.fillText(s.id, x, y);
  }
  // Duplicate lines (magenta between slot pairs on same GT).
  for (const cov of gtCovered) {
    if (cov.length >= 2) {
      for (let i = 0; i < cov.length; i++) for (let j = i + 1; j < cov.length; j++) {
        const [x1, y1] = posToPx(cov[i].pos);
        const [x2, y2] = posToPx(cov[j].pos);
        ctx3.strokeStyle = "magenta"; ctx3.lineWidth = 2;
        ctx3.beginPath(); ctx3.moveTo(x1, y1); ctx3.lineTo(x2, y2); ctx3.stroke();
      }
    }
  }
  document.getElementById("info_cov").textContent =
    "GT " + aliveGT.length + " | pred alive " + aliveSlots.length +
    " | missed " + missed + " | dup " + dup + " | phantom " + phantom +
    " (radius " + R + " normalized)";

  // ---- Canvas 4: reconstruction ----
  const c4 = document.getElementById("canvas_recon");
  const ctx4 = c4.getContext("2d");
  ctx4.imageSmoothingEnabled = false;
  if (state.iter === 0) {
    // No composite for iter 0 (initial state, no forward pass done).
    ctx4.fillStyle = "#eee"; ctx4.fillRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    ctx4.fillStyle = "#666"; ctx4.font = "16px sans-serif"; ctx4.textAlign = "center";
    ctx4.fillText("(iter 0 — no composite computed)", DISPLAY_SIZE / 2, DISPLAY_SIZE / 2);
    document.getElementById("info_recon").textContent = "iter 0 — initial state";
  } else {
    const comp = composites[state.iter - 1];
    ctx4.drawImage(comp, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    document.getElementById("info_recon").textContent =
      "reconstruction at iter " + state.iter +
      (state.iter === DATA.trained_iters ? "  (trained)" : "");
  }

  // State info.
  const alive_count = zps.filter(v => v > 0.5).length;
  document.getElementById("info_state").innerHTML =
    "alive slots: " + alive_count + " / " + DATA.n_slots + "<br>" +
    "z_pres: [" + zps.map(v => v.toFixed(2)).join(", ") + "]";
}
</script>
</body>
</html>
""".replace("__PAYLOAD__", payload_json)


if __name__ == "__main__":
    main()
