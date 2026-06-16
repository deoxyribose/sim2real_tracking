"""Lightweight checkpoint helpers — save and restore Flax params + opt_state to disk.

Uses pickle for simplicity. Orbax adds robustness but introduces a heavy dep we don't need yet.
"""

from __future__ import annotations

import os
import pickle


def save(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
