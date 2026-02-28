from __future__ import annotations
import itertools
from typing import List, Tuple

from .experiment import run_one_experiment
from ..core.utils import append_jsonl


def ring_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def main() -> None:
    out_path = "data.jsonl"

    seeds = [0, 1, 2, 3, 4]
    shots = 1024

    p2_values = [0.0, 0.002, 0.005, 0.01, 0.02]
    p1_values = [0.0, 0.001, 0.003]
    pm_values = [0.0, 0.01]

    # ----------------
    # Grover sweep
    # ----------------
    for n in [6, 8]:
        marked = 5
        k_values = list(range(0, 10))  # keep small; depth grows fast with MCX

        for k, p1, pm, seed in itertools.product(k_values, p1_values, pm_values, seeds):
            spec = {
                "algorithm": "grover",
                "seed": seed,
                "noise": {"p1": p1, "p2": 0.0, "pm": pm},
                "algo_cfg": {
                    "n": n,
                    "marked": marked,
                    "iterations": k,
                    "shots": shots,
                    "seed": seed,
                },
            }
            row = run_one_experiment(spec)
            append_jsonl(out_path, row)

    # ----------------
    # QAOA MaxCut sweep
    # ----------------
    for n in [6, 8]:
        edges = ring_edges(n)
        for p in [1, 2, 3]:
            gammas = [0.5] * p
            betas = [0.7] * p

            for p2, p1, pm, seed in itertools.product(p2_values, p1_values, pm_values, seeds):
                spec = {
                    "algorithm": "qaoa_maxcut",
                    "seed": seed,
                    "noise": {"p1": p1, "p2": p2, "pm": pm},
                    "algo_cfg": {
                        "n": n,
                        "edges": edges,
                        "p": p,
                        "gammas": gammas,
                        "betas": betas,
                        "shots": shots,
                        "seed": seed,
                    },
                }
                row = run_one_experiment(spec)
                append_jsonl(out_path, row)

    print(f"Done. Wrote dataset to {out_path}")


if __name__ == "__main__":
    main()
