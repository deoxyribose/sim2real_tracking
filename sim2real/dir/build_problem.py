"""Build the CP-SAT ILP problem from a hypothesis pool.

Simplified from `/home/frans/discrete_linking_opt/2_make_discrete_opt_problem_celegans.py`.
Kept: node reconstruction costs, same-frame at-most-one on overlapping pairs,
event-flow tracking (birth/death/link) with skeleton-distance link costs and
gap-linking across small blackouts. Dropped: multilinear overlap interactions
(we let the at-most-one handle it), cell divisions (flagella don't split).

A `Hypothesis` bundles a skeleton, width, amp, score, and its precomputed
rendered contribution to the frame. `build_problem(...)` returns a data dict
consumable by `solve.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np

from .renderer import hypothesis_mask, render_flagellum


@dataclass
class Hypothesis:
    frame: int                      # 0..T-1
    skeleton: np.ndarray            # (K, 2)  world (y, x)
    width: float                    # px
    amp: float                      # signed intensity
    score: float                    # model confidence in [0, 1]
    # populated by build_problem:
    rendered: np.ndarray | None = None   # (H, W) signed
    support: np.ndarray | None = None    # (H, W) bool
    node_cost: int = 0                   # unary cost (integer, scaled)


@dataclass
class BuildConfig:
    # Cost scaling — we work in integer units (CP-SAT prefers ints).
    # cost_scale multiplies float costs before rounding.
    cost_scale: float = 100.0
    # Two modes for the node cost:
    #   "recon+score" : reconstruction-error delta − score_bonus·score
    #   "score_only"  : constant per-pick − score_bonus·score
    # Use "score_only" until the model's rendered predictions are accurate
    # enough that reconstruction gains outweigh their errors.
    cost_mode: str = "score_only"
    # Fixed cost per selected hypothesis in `score_only` mode. Should be small
    # enough that any reasonably scored hypothesis is picked.
    pick_cost_base: float = 1.0
    # Same-frame overlap: two hypotheses whose supports overlap by more than
    # this fraction of the smaller support → at-most-one.
    max_pair_overlap_frac: float = 0.4
    # Score bonus: subtract score_bonus_per_unit_score * score from node cost
    # so high-confidence predictions are cheaper (rewarded).
    score_bonus: float = 100.0
    # Temporal linking
    link_max_gap: int = 2                # bridge up to N missing frames
    link_max_dist: float = 30.0          # px — no link beyond this skeleton dist
    link_cost_scale: float = 1.0         # per-px cost of a link
    link_gap_cost_factor: float = 1.5    # multiply cost by factor**(gap-1)
    # Track transitions
    birth_cost: float = 20.0             # constant per-track birth
    death_cost: float = 20.0             # constant per-track death


def _sym_skeleton_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-vertex L2 between two skeletons; accounts for reverse traversal."""
    d1 = float(np.linalg.norm(a - b, axis=-1).mean())
    d2 = float(np.linalg.norm(a - b[::-1], axis=-1).mean())
    return min(d1, d2)


def build_problem(hypotheses: list[Hypothesis],
                  residuals: np.ndarray,      # (T, H, W) float32 signed
                  cfg: BuildConfig = BuildConfig()) -> dict:
    """Build the ILP data.

    Args:
      hypotheses: flat list; must include `.frame` and skeleton/width/amp/score.
                  Rendered fields will be filled in-place.
      residuals:  (T, H, W) median-subtracted residual, in the same intensity
                  units as `hypothesis.amp`.
    Returns:
      dict with keys:
        num_variables, node_costs, at_most_one_constraints,
        links=list of (src, dst, cost, gap), birth_costs, death_costs, cfg
    """
    T, H, W = residuals.shape
    N = len(hypotheses)

    # -- Render each hypothesis + compute unary cost -----------------------
    # Unary cost = increase in reconstruction L1 when we pick this hypothesis
    # in isolation. reference frame is "no hypotheses selected", i.e. render=0.
    for h in hypotheses:
        h.rendered = render_flagellum(h.skeleton, h.width, h.amp, H, W)
        h.support = hypothesis_mask(h.skeleton, h.width, H, W)

    residual_norms = np.abs(residuals).sum(axis=(1, 2))         # (T,) per-frame

    node_costs_int: list[int] = []
    for h in hypotheses:
        if cfg.cost_mode == "score_only":
            cost = cfg.pick_cost_base - cfg.score_bonus * h.score
        elif cfg.cost_mode == "recon+score":
            r = residuals[h.frame]
            m = h.support
            base = float(np.abs(r[m]).sum())
            after = float(np.abs(r[m] - h.rendered[m]).sum())
            raw = after - base                                   # negative if beneficial
            cost = raw - cfg.score_bonus * h.score
        else:
            raise ValueError(f"unknown cost_mode: {cfg.cost_mode!r}")
        node_costs_int.append(int(round(cfg.cost_scale * cost)))
        h.node_cost = node_costs_int[-1]

    # -- Same-frame overlap → at-most-one on offending pairs --------------
    by_frame: dict[int, list[int]] = {}
    for i, h in enumerate(hypotheses):
        by_frame.setdefault(h.frame, []).append(i)

    at_most_one: list[list[int]] = []
    for frame_idx, ids in by_frame.items():
        for a, b in combinations(ids, 2):
            ha = hypotheses[a]; hb = hypotheses[b]
            ia = int((ha.support & hb.support).sum())
            if ia == 0:
                continue
            frac = ia / max(1, min(int(ha.support.sum()), int(hb.support.sum())))
            if frac > cfg.max_pair_overlap_frac:
                at_most_one.append([a, b])

    # -- Temporal links ---------------------------------------------------
    links: list[tuple[int, int, int, int]] = []   # (src, dst, cost, gap)
    for gap in range(1, cfg.link_max_gap + 1):
        for t_src in range(T - gap):
            t_dst = t_src + gap
            src_ids = by_frame.get(t_src, [])
            dst_ids = by_frame.get(t_dst, [])
            if not src_ids or not dst_ids:
                continue
            for i in src_ids:
                for j in dst_ids:
                    d = _sym_skeleton_dist(hypotheses[i].skeleton,
                                             hypotheses[j].skeleton)
                    if d > cfg.link_max_dist:
                        continue
                    cost = cfg.link_cost_scale * d * (cfg.link_gap_cost_factor ** (gap - 1))
                    links.append((i, j, int(round(cfg.cost_scale * cost)), gap))

    # -- Birth/death costs (constant, per hypothesis) ---------------------
    birth_cost_int = int(round(cfg.cost_scale * cfg.birth_cost))
    death_cost_int = int(round(cfg.cost_scale * cfg.death_cost))
    birth_costs = [birth_cost_int] * N
    death_costs = [death_cost_int] * N

    return dict(
        num_variables=N,
        node_costs=node_costs_int,
        at_most_one_constraints=at_most_one,
        links=links,
        birth_costs=birth_costs,
        death_costs=death_costs,
        by_frame=by_frame,
        cfg=cfg,
        residual_norms=residual_norms,
    )
