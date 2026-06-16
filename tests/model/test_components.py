"""Smoke tests for encoder, STN, and glimpse modules."""

import jax
import jax.numpy as jnp

from sim2real.model.encoder import FrameEncoder
from sim2real.model.glimpse import GlimpseDecoder, GlimpseEncoder, SegHead
from sim2real.model.posenc import sinusoidal_2d
from sim2real.model.stn import stn_read, stn_write


def test_encoder_shape():
    enc = FrameEncoder(d_model=128, n_vit_layers=1)
    img = jnp.zeros((64, 64, 1))
    key = jax.random.key(0)
    params = enc.init(key, img)
    fg, pool = enc.apply(params, img)
    assert fg.shape == (8, 8, 128)
    assert pool.shape == (128,)


def test_posenc():
    pe = sinusoidal_2d(8, 12, 64)
    assert pe.shape == (8, 12, 64)
    # Non-trivial values
    assert float(pe.std()) > 0.1


def test_stn_read_write_shapes():
    img = jnp.ones((32, 32, 1))
    zw = jnp.array([-0.5, 0.0, 0.0])  # smaller scale
    g = stn_read(img, zw, 16)
    assert g.shape == (16, 16, 1)
    out = stn_write(g, zw, 32)
    assert out.shape == (32, 32, 1)


def test_stn_read_recovers_constant():
    # A constant image read at any z_where should produce a constant glimpse.
    img = 0.7 * jnp.ones((32, 32, 1))
    zw = jnp.array([0.0, 0.2, -0.3])
    g = stn_read(img, zw, 8)
    # Interior pixels should be ≈ 0.7 (boundary may be 0 if outside).
    assert float(g[3, 3, 0]) > 0.6


def test_glimpse_decoder_shapes():
    gd = GlimpseDecoder(glimpse_size=16, z_what_dim=32)
    key = jax.random.key(0)
    params = gd.init(key, jnp.zeros((32,)))
    appear, m = gd.apply(params, jnp.zeros((32,)))
    assert appear.shape == (16, 16, 1)
    assert m.shape == (16, 16, 1)
    assert 0.0 <= float(appear.min()) and float(appear.max()) <= 1.0


def test_seg_head_shape():
    sh = SegHead(glimpse_size=16, hidden=64)
    key = jax.random.key(0)
    params = sh.init(key, jnp.zeros((32,)), jnp.zeros((3,)))
    m = sh.apply(params, jnp.zeros((32,)), jnp.zeros((3,)))
    assert m.shape == (16, 16, 1)


def test_glimpse_encoder_shape():
    ge = GlimpseEncoder(feat_dim=64)
    key = jax.random.key(0)
    params = ge.init(key, jnp.zeros((16, 16, 1)))
    f = ge.apply(params, jnp.zeros((16, 16, 1)))
    assert f.shape == (64,)


if __name__ == "__main__":
    import inspect, sys
    failed = 0
    for name, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}")
            failed += 1
    sys.exit(failed)
