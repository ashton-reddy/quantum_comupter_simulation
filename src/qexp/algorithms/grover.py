from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
import numpy as np


@dataclass(frozen=True)
class GroverConfig:
    n: int
    marked: int
    iterations: int
    shots: int
    seed: int = 0


def _apply_phase_oracle(qc: QuantumCircuit, marked: int) -> None:
    """
    Flip the phase of |marked>.
    Implementation:
      - For bits that are 0 in marked: X (so target becomes all-ones)
      - Multi-controlled Z on last qubit via MCX + H trick
      - Undo X
    """
    n = qc.num_qubits
    bits = [(marked >> i) & 1 for i in range(n)]  # qubit 0 is LSB
    zero_qubits = [i for i, b in enumerate(bits) if b == 0]

    for q in zero_qubits:
        qc.x(q)

    # Controlled-Z on |11..1> (controls: 0..n-2, target: n-1)
    qc.h(n - 1)
    qc.append(MCXGate(num_ctrl_qubits=n - 1), list(range(n)))  # controls + target
    qc.h(n - 1)

    for q in zero_qubits:
        qc.x(q)


def _apply_diffuser(qc: QuantumCircuit) -> None:
    """
    Diffusion operator: 2|s><s| - I
    Standard construction with H, X, and multi-controlled Z.
    """
    n = qc.num_qubits
    qc.h(range(n))
    qc.x(range(n))

    qc.h(n - 1)
    qc.append(MCXGate(num_ctrl_qubits=n - 1), list(range(n)))
    qc.h(n - 1)

    qc.x(range(n))
    qc.h(range(n))


def build_grover_circuit(cfg: GroverConfig) -> QuantumCircuit:
    qc = QuantumCircuit(cfg.n, cfg.n)
    qc.h(range(cfg.n))

    for _ in range(cfg.iterations):
        _apply_phase_oracle(qc, cfg.marked)
        _apply_diffuser(qc)

    qc.measure(range(cfg.n), range(cfg.n))
    return qc


def grover_metrics(counts: Dict[str, int], cfg: GroverConfig) -> Dict[str, Any]:
    marked_str = format(cfg.marked, f"0{cfg.n}b")
    success = counts.get(marked_str, 0) / max(1, sum(counts.values()))
    return {"success_prob": float(success), "marked_bitstring": marked_str}
