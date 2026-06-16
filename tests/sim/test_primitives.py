"""Smoke tests for sim primitives — shape, range, vmap, jit."""

import jax
import jax.numpy as jnp

from sim2real.sim.background import perlin_background, perlin_noise
from sim2real.sim.primitives import alpha_composite_with_bg, gaussian_blob, make_grid
from sim2real.sim.render_common import composite_video_frame, gaussian_blob_mask, perlin_grayscale_bg
from sim2real.sim.splines import (
    arc_length_resample,
    catmull_rom_eval,
    knots2params,
    params2knots,
    render_polyline_segments_sdf,
)


def test_grid_blob_range():
    xx, yy = make_grid(32)
    assert xx.shape == (32, 32)
    assert float(xx.min()) == -1.0 and float(xx.max()) == 1.0
    b = gaussian_blob(xx, yy, 0.0, 0.0, 0.2)
    assert 0.0 <= float(b.min()) and float(b.max()) <= 1.0001


def test_perlin_shape_and_traceable():
    key = jax.random.key(0)
    p = perlin_noise(key, 32, steps=4)
    assert p.shape == (32, 32)

    # JIT and vmap should both work.
    jp = jax.jit(lambda k: perlin_background(k, 32))(key)
    assert jp.shape == (32, 32)

    vk = jax.random.split(key, 3)
    vp = jax.vmap(lambda k: perlin_background(k, 32))(vk)
    assert vp.shape == (3, 32, 32)


def test_perlin_range_in_unit_interval():
    key = jax.random.key(1)
    bg = perlin_background(key, 32)
    assert float(bg.min()) >= 0.0
    assert float(bg.max()) <= 1.0


def test_splines_roundtrip_and_render():
    p = jnp.array([-1.5, 0.0, 1.5])
    assert jnp.allclose(knots2params(params2knots(p)), p, atol=1e-5)

    ctrl = jnp.array([[-0.5, 0.0], [0.0, 0.4], [0.4, -0.2]])
    curve = catmull_rom_eval(ctrl, 20)
    assert curve.shape == (20, 2)
    res = arc_length_resample(curve, 10)
    assert res.shape == (10, 2)
    sdf = render_polyline_segments_sdf(res, 32, sigma=1.0)
    assert sdf.shape == (32, 32) and float(sdf.max()) > 0.5


def test_composite_with_bg():
    bg = perlin_grayscale_bg(jax.random.key(0), 32, 1)
    m = gaussian_blob_mask(32, 0.0, 0.0, 0.2)
    masks = jnp.stack([m])
    appear = masks[..., None] * jnp.array([[1.0]])[:, None, None, :]
    z_pres = jnp.array([1.0])
    out = composite_video_frame(masks, appear, bg, z_pres)
    assert out.shape == (32, 32, 1)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


if __name__ == "__main__":
    import inspect
    import sys

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
