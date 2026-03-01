from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# -----------------------------
# Shared building blocks
# -----------------------------

def diffusion(qc: QuantumCircuit, qubits: List[int]) -> None:
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)

    qc.h(qubits[-1])
    if n == 1:
        qc.z(qubits[0])
    elif n == 2:
        qc.cx(qubits[0], qubits[1])
    else:
        qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])

    qc.x(qubits)
    qc.h(qubits)


def phase_oracle_marked_state(qc: QuantumCircuit, qubits: List[int], marked: str) -> None:
    if len(marked) != len(qubits) or any(b not in "01" for b in marked):
        raise ValueError("marked must be a bitstring of length n containing only 0/1")

    for i, bit in enumerate(marked):
        if bit == "0":
            qc.x(qubits[i])

    qc.h(qubits[-1])
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cx(qubits[0], qubits[1])
    else:
        qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])

    for i, bit in enumerate(marked):
        if bit == "0":
            qc.x(qubits[i])


# -----------------------------
# NEW API used by sweep.py
# -----------------------------

@dataclass(frozen=True)
class GroverSpec:
    n: int
    marked: str      # bitstring length n
    iters: int
    shots: int


def build_grover(spec: GroverSpec) -> QuantumCircuit:
    q = QuantumRegister(spec.n, "q")
    c = ClassicalRegister(spec.n, "c")
    qc = QuantumCircuit(q, c)

    qc.h(q)
    for _ in range(spec.iters):
        phase_oracle_marked_state(qc, list(range(spec.n)), spec.marked)
        diffusion(qc, list(range(spec.n)))

    qc.measure(q, c)
    qc.name = f"grover_n{spec.n}_k{spec.iters}_m{spec.marked}"
    return qc


def grover_metrics(counts: Dict[str, int], spec: GroverSpec) -> Dict[str, Any]:
    shots = max(1, sum(counts.values()))
    return {"success_prob": float(counts.get(spec.marked, 0) / shots)}


# -----------------------------
# OLD API used by experiment.py
# -----------------------------

@dataclass(frozen=True)
class GroverConfig:
    n: int
    marked: int        # integer marked state
    iterations: int
    shots: int
    seed: int = 0


def build_grover_circuit(cfg: GroverConfig) -> QuantumCircuit:
    # Convert int -> bitstring (MSB..LSB)
    marked_str = format(cfg.marked, f"0{cfg.n}b")
    spec = GroverSpec(n=cfg.n, marked=marked_str, iters=cfg.iterations, shots=cfg.shots)
    return build_grover(spec)


def grover_metrics_legacy(counts: Dict[str, int], cfg: GroverConfig) -> Dict[str, Any]:
    marked_str = format(cfg.marked, f"0{cfg.n}b")
    shots = max(1, sum(counts.values()))
    return {"success_prob": float(counts.get(marked_str, 0) / shots), "marked_bitstring": marked_str}


# Keep the old name used in your experiment.py import
# (experiment.py expects grover_metrics(counts, cfg))
def grover_metrics(counts: Dict[str, int], cfg_or_spec) -> Dict[str, Any]:
    if isinstance(cfg_or_spec, GroverSpec):
        return grover_metrics(counts, cfg_or_spec)
    # fallback legacy
    return grover_metrics_legacy(counts, cfg_or_spec)

# -----------------------------
# Legacy helpers (experiment.py)
# -----------------------------

@dataclass(frozen=True)
class GroverConfig:
    n: int
    marked: int
    iterations: int
    shots: int
    seed: int = 0


def build_grover_circuit(cfg: GroverConfig) -> QuantumCircuit:
    marked_str = format(cfg.marked, f"0{cfg.n}b")
    spec = GroverSpec(n=cfg.n, marked=marked_str, iters=cfg.iterations, shots=cfg.shots)
    return build_grover(spec)


def grover_metrics_legacy(counts: Dict[str, int], cfg: GroverConfig) -> Dict[str, Any]:
    marked_str = format(cfg.marked, f"0{cfg.n}b")
    shots = max(1, sum(counts.values()))
    return {"success_prob": float(counts.get(marked_str, 0) / shots), "marked_bitstring": marked_str}


# Keep a single exported name that works for both
def grover_metrics(counts: Dict[str, int], cfg_or_spec) -> Dict[str, Any]:
    if isinstance(cfg_or_spec, GroverSpec):
        return {"success_prob": float(counts.get(cfg_or_spec.marked, 0) / max(1, sum(counts.values())))}
    return grover_metrics_legacy(counts, cfg_or_spec)