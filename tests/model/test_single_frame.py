"""Single-frame composition test: encoder → slot transformer → heads → decoder → composite.

This validates that all components plug together and produce a finite recon loss. The full
multi-frame model lives in `sim2real/model/model.py` (next task).
"""

import jax
import jax.numpy as jnp

from sim2real.model.encoder import FrameEncoder
from sim2real.model.glimpse import GlimpseDecoder, GlimpseEncoder
from sim2real.model.heads import PresHead, WhatHead, WhereHead
from sim2real.model.slot_transformer import SlotTransformer
from sim2real.model.stn import stn_read, stn_write


def _build_modules(d=64, n_max=8, z_what_dim=32, glimpse=12, res=64):
    enc = FrameEncoder(d_model=d, n_vit_layers=1, stem_channels=(16, 32, 48))
    st = SlotTransformer(n_max=n_max, d_model=d, n_heads=4, n_layers=2)
    ge = GlimpseEncoder(feat_dim=z_what_dim, channels=(16, 32))
    where_h = WhereHead(scale=0.5, hidden=32)
    pres_h = PresHead(hidden=32, init_bias=-1.0)
    what_h = WhatHead(z_what_dim=z_what_dim, hidden=32)
    gd = GlimpseDecoder(glimpse_size=glimpse, z_what_dim=z_what_dim, channels=(32, 16))
    return enc, st, ge, where_h, pres_h, what_h, gd


def test_single_frame_forward_finite_loss():
    res, d, n_max, z_what_dim, glimpse = 64, 64, 8, 32, 12
    enc, st, ge, where_h, pres_h, what_h, gd = _build_modules(d, n_max, z_what_dim, glimpse, res)

    key = jax.random.key(0)
    img = jax.random.uniform(jax.random.fold_in(key, 0), (res, res, 1))

    # Init each module separately.
    enc_p = enc.init(jax.random.fold_in(key, 1), img)
    fg, _ = enc.apply(enc_p, img)
    st_p = st.init(jax.random.fold_in(key, 2), fg, jnp.zeros((n_max, d)))
    q_prop, q_disc, _ = st.apply(st_p, fg, jnp.zeros((n_max, d)))

    # Heads (vmapped over slots).
    where_p = where_h.init(jax.random.fold_in(key, 3), q_prop[0], jnp.zeros((5,)))
    pres_p = pres_h.init(jax.random.fold_in(key, 4), q_prop[0], key)
    what_p = what_h.init(jax.random.fold_in(key, 5), q_prop[0], jnp.zeros((z_what_dim,)), key)
    ge_p = ge.init(jax.random.fold_in(key, 6), jnp.zeros((glimpse, glimpse, 1)))
    gd_p = gd.init(jax.random.fold_in(key, 7), jnp.zeros((z_what_dim,)))

    prev_zwhere = jnp.zeros((n_max, 5))

    def per_slot(q, k_p, k_w, prev_zw):
        zwhere = where_h.apply(where_p, q, prev_zw)
        zpres, _ = pres_h.apply(pres_p, q, k_p)
        # Read glimpse from image at z_where
        gl = stn_read(img, zwhere, glimpse)
        glimpse_feat = ge.apply(ge_p, gl)
        zwhat, _mu, _lv = what_h.apply(what_p, q, glimpse_feat, k_w)
        # Decode patch
        appear_patch, mask_logit_patch = gd.apply(gd_p, zwhat)
        mask = jax.nn.sigmoid(mask_logit_patch) * zpres
        appear_canvas = stn_write(appear_patch, zwhere, res)                        # (R, R, 1)
        mask_canvas = stn_write(mask, zwhere, res)[..., 0]                          # (R, R)
        return appear_canvas, mask_canvas

    keys_p = jax.random.split(jax.random.fold_in(key, 8), n_max)
    keys_w = jax.random.split(jax.random.fold_in(key, 9), n_max)
    appear, masks = jax.vmap(per_slot)(q_prop, keys_p, keys_w, prev_zwhere)         # (N, R, R, 1), (N, R, R)

    # Composite over (no background — just slots).
    num = jnp.sum(appear * masks[..., None], axis=0)
    den = jnp.sum(masks, axis=0)[..., None] + 1e-6
    composite = jnp.clip(num / den, 0.0, 1.0)                                       # (R, R, 1)
    loss = jnp.mean((composite - img) ** 2)
    assert jnp.isfinite(loss)
    assert loss.shape == ()
    assert float(loss) < 1.0


if __name__ == "__main__":
    test_single_frame_forward_finite_loss()
    print("PASS test_single_frame_forward_finite_loss")
