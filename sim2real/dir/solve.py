"""CP-SAT solve of the DIR problem.

Adapted (simplified) from `/home/frans/discrete_linking_opt/3_solve_discrete_problem.py`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from ortools.sat.python import cp_model


@dataclass
class SolveConfig:
    time_limit_s: float = 60.0
    num_workers: int = 8
    log_search: bool = False


def solve_problem(problem: dict, cfg: SolveConfig = SolveConfig()) -> dict:
    """Solve the ILP built by `build_problem`.

    Returns:
      dict with keys: status, objective, selected_indices, links_selected,
                        tracks (list[list[hypothesis_id]]).
    """
    N = int(problem["num_variables"])
    node_costs = problem["node_costs"]
    at_most_one = problem["at_most_one_constraints"]
    links = problem["links"]                                  # list[(src, dst, cost, gap)]
    birth_costs = problem["birth_costs"]
    death_costs = problem["death_costs"]

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x[{i}]") for i in range(N)]
    b = [model.NewBoolVar(f"b[{i}]") for i in range(N)]
    d = [model.NewBoolVar(f"d[{i}]") for i in range(N)]
    y = [model.NewBoolVar(f"y[{e}]") for e in range(len(links))]

    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for e, (src, dst, _c, _g) in enumerate(links):
        outgoing[int(src)].append(e)
        incoming[int(dst)].append(e)

    # At-most-one constraints
    for group in at_most_one:
        model.AddAtMostOne(x[int(i)] for i in group)

    # Link implications: link only if both endpoints selected
    for e, (src, dst, _c, _g) in enumerate(links):
        model.Add(y[e] <= x[int(src)])
        model.Add(y[e] <= x[int(dst)])

    # Flow conservation: x = birth + sum(incoming) = death + sum(outgoing)
    for i in range(N):
        model.Add(b[i] + sum(y[e] for e in incoming[i]) == x[i])
        model.Add(d[i] + sum(y[e] for e in outgoing[i]) == x[i])

    # Objective
    terms = []
    for i, c in enumerate(node_costs):
        if int(c) != 0:
            terms.append(int(c) * x[i])
    for i, c in enumerate(birth_costs):
        if int(c) != 0:
            terms.append(int(c) * b[i])
    for i, c in enumerate(death_costs):
        if int(c) != 0:
            terms.append(int(c) * d[i])
    for e, (_src, _dst, c, _g) in enumerate(links):
        if int(c) != 0:
            terms.append(int(c) * y[e])
    model.Minimize(sum(terms) if terms else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(cfg.time_limit_s)
    solver.parameters.num_search_workers = int(cfg.num_workers)
    solver.parameters.log_search_progress = bool(cfg.log_search)

    status = solver.Solve(model)
    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, str(status))

    selected = [i for i in range(N) if solver.Value(x[i]) == 1]
    link_sel_pairs = [(int(links[e][0]), int(links[e][1]))
                      for e in range(len(links)) if solver.Value(y[e]) == 1]
    births = [i for i in range(N) if solver.Value(b[i]) == 1]
    deaths = [i for i in range(N) if solver.Value(d[i]) == 1]

    # Assemble tracks by walking forward from each birth
    src_to_dst = {}
    for (s, t) in link_sel_pairs:
        src_to_dst.setdefault(s, []).append(t)
    tracks: list[list[int]] = []
    for start in births:
        track = [start]
        cur = start
        while cur in src_to_dst:
            nxt = src_to_dst[cur][0]     # branching disallowed for flagella
            track.append(nxt)
            cur = nxt
        tracks.append(track)

    return dict(
        status=status_name,
        objective=solver.ObjectiveValue(),
        best_bound=solver.BestObjectiveBound(),
        wall_time=solver.WallTime(),
        selected_indices=selected,
        links_selected=link_sel_pairs,
        births=births,
        deaths=deaths,
        tracks=tracks,
    )
