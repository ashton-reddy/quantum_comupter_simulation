from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


@dataclass(frozen=True)
class QECSpec:
    shots: int
    cycles: int
    seed: int = 0


def build_bitflip_syndrome_round() -> QuantumCircuit:
    """
    3 data qubits + 2 ancillas.
    Measures parity checks (stabilizers) Z0Z1 and Z1Z2 using ancillas.
    The 2 classical bits are the syndrome (ancilla measurement outcomes).
    """
    d = QuantumRegister(3, "d")
    a = QuantumRegister(2, "a")
    s = ClassicalRegister(2, "s")
    qc = QuantumCircuit(d, a, s)

    # Prepare (|000> + |111>)/sqrt2 as a simple entangled state to test detection
    qc.h(d[0])
    qc.cx(d[0], d[1])
    qc.cx(d[0], d[2])

    # Measure Z0Z1 on ancilla a0
    qc.cx(d[0], a[0])
    qc.cx(d[1], a[0])
    qc.measure(a[0], s[0])

    # Measure Z1Z2 on ancilla a1
    qc.cx(d[1], a[1])
    qc.cx(d[2], a[1])
    qc.measure(a[1], s[1])

    qc.name = "bitflip_syndrome_round"
    return qc


def syndrome_metrics(counts: Dict[str, int]) -> Dict[str, Any]:
    """
    counts keys are 2-bit strings (syndrome bits). Example: '00','01','10','11'
    """
    shots = max(1, sum(counts.values()))
    probs = {k: v / shots for k, v in counts.items()}

    # error rate per cycle: any non-00 syndrome
    err_rate = 1.0 - probs.get("00", 0.0)

    # correlation between syndrome bits s0 and s1
    # Here, syndrome string is like 'ab' where a=s[0], b=s[1]
    def bit(s: str, idx: int) -> int:
        return int(s[idx])

    es0 = sum(bit(k, 0) * probs[k] for k in probs)
    es1 = sum(bit(k, 1) * probs[k] for k in probs)
    es0s1 = sum(bit(k, 0) * bit(k, 1) * probs[k] for k in probs)
    corr = es0s1 - es0 * es1

    return {
        "syndrome_probs": probs,
        "error_rate_this_cycle": float(err_rate),
        "syndrome_bit_correlation": float(corr),
        "ancilla_qubit_measurements": "syndrome bits s0,s1",
        "stabilizer_outcomes": "syndrome distribution over {00,01,10,11}",
        "parity_checks": ["Z0Z1", "Z1Z2"],
    }