from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class QAOAConfig:
    n: int
    edges: List[Tuple[int, int]]
    p: int
    gammas: List[float]
    betas: List[float]
    shots: int
    seed: int = 0


def cut_value(bitstring: str, edges: List[Tuple[int, int]]) -> int:
    # bitstring is MSB..LSB in Qiskit counts keys.
    # Convert to index by position: qubit i corresponds to bitstring[-1 - i]
    def b(i: int) -> int:
        return int(bitstring[-1 - i])
    return sum(b(u) ^ b(v) for (u, v) in edges)


def build_qaoa_maxcut_circuit(cfg: QAOAConfig) -> QuantumCircuit:
    assert len(cfg.gammas) == cfg.p and len(cfg.betas) == cfg.p

    qc = QuantumCircuit(cfg.n, cfg.n)
    qc.h(range(cfg.n))

    for layer in range(cfg.p):
        gamma = cfg.gammas[layer]
        beta = cfg.betas[layer]

        # cost: apply RZZ(2*gamma) on each edge
        for (u, v) in cfg.edges:
            qc.rzz(2.0 * gamma, u, v)

        # mixer: RX(2*beta) on each qubit
        for q in range(cfg.n):
            qc.rx(2.0 * beta, q)

    qc.measure(range(cfg.n), range(cfg.n))
    return qc


def qaoa_metrics(counts: Dict[str, int], cfg: QAOAConfig) -> Dict[str, Any]:
    shots = max(1, sum(counts.values()))
    # sample mean cut from measurement distribution
    mean_cut = sum(cut_value(bs, cfg.edges) * c for bs, c in counts.items()) / shots
    # best observed bitstring in counts
    best_bs = max(counts.items(), key=lambda kv: cut_value(kv[0], cfg.edges))[0]
    best_cut = cut_value(best_bs, cfg.edges)
    best_prob = counts[best_bs] / shots
    return {
        "sample_mean_cut": float(mean_cut),
        "best_observed_bitstring": best_bs,
        "best_observed_cut": int(best_cut),
        "best_observed_prob": float(best_prob),
    }
