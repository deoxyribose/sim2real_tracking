"""Path-glob-based parameter freezing using `optax.masked`.

`build_freeze_mask(params, freeze_patterns) -> mask` returns a pytree of bools (True = frozen)
that can be passed to `optax.masked(transform, mask)` or used directly with `optax.set_to_zero`.

Patterns match the dotted path through the params pytree, e.g.:
    ["encoder.*"]         freezes everything inside the encoder.
    ["slot_transformer.*"] freezes the slot transformer.
    ["*"]                  freezes everything (no-op training).
"""

from __future__ import annotations

import fnmatch
from typing import Sequence

import jax


def _path_str(path) -> str:
    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(p.key)
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    # Strip the leading "params" container that flax inserts.
    if parts and parts[0] == "params":
        parts = parts[1:]
    return ".".join(parts)


def build_freeze_mask(params, patterns: Sequence[str]):
    """Return a pytree of bools, True where the parameter path matches any of `patterns`."""

    def mark(path, _leaf):
        p = _path_str(path)
        return any(fnmatch.fnmatch(p, pat) for pat in patterns)

    return jax.tree_util.tree_map_with_path(mark, params)


def make_optimizer_with_freeze(base_opt, params, freeze_patterns: Sequence[str]):
    """Wrap an optimizer so that frozen params receive zero updates.

    Implemented via `optax.masked(set_to_zero, mask_of_frozen) + masked(base_opt, mask_of_trainable)`
    — we just return `optax.chain` of two masked transforms. With `optax.masked`, `mask=True`
    selects which leaves the transform applies to.
    """
    import optax

    mask_frozen = build_freeze_mask(params, freeze_patterns)

    def invert(mask):
        return jax.tree.map(lambda x: not bool(x), mask)

    mask_train = invert(mask_frozen)
    opt = optax.chain(
        optax.masked(optax.set_to_zero(), mask_frozen),
        optax.masked(base_opt, mask_train),
    )
    return opt, mask_frozen
