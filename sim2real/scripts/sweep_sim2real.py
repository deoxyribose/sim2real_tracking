"""Sim2real experiment orchestrator.

Given a pretrained checkpoint, run adapt on a list of (sim_kind, sim_cfg) target settings; for
each, write eval metrics to `runs/sim2real/<expname>/metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def run(cmd):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    return res.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, help="path to a pretrain ckpt")
    ap.add_argument("--sims", nargs="+", default=["flagella", "many_cells"])
    ap.add_argument("--adapt-steps", type=int, default=200)
    ap.add_argument("--out", default="runs/sim2real")
    args = ap.parse_args()

    py = sys.executable
    os.environ["PYTHONPATH"] = "."
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    for sim in args.sims:
        adapt_dir = os.path.join(args.out, f"adapt_{sim}")
        adapt_cmd = [
            py, "-m", "sim2real.scripts.adapt",
            "--sim", sim,
            "--pretrain-ckpt", args.pretrained,
            "--steps", str(args.adapt_steps),
            "--run-dir", adapt_dir,
            "--freeze", "encoder.*",
        ]
        if run(adapt_cmd) != 0:
            print(f"adapt failed for {sim}")
            continue

        # find latest ckpt
        ckpts_dir = os.path.join(adapt_dir, "ckpts")
        latest = sorted(os.listdir(ckpts_dir))[-1]
        ckpt_path = os.path.join(ckpts_dir, latest)
        metrics_path = os.path.join(adapt_dir, "metrics.json")

        eval_cmd = [
            py, "-m", "sim2real.scripts.eval_ckpt",
            "--ckpt", ckpt_path,
            "--sim", sim,
            "--out", metrics_path,
        ]
        run(eval_cmd)
        try:
            with open(metrics_path) as f:
                summary[sim] = json.load(f)
        except Exception:
            pass

    out_summary = os.path.join(args.out, "summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_summary}")


if __name__ == "__main__":
    main()
