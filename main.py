from __future__ import annotations

import math
from typing import List, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.basic_provider import BasicSimulator


def diffusion(qc: QuantumCircuit, qubits: List[int]) -> None:
    """
    Inversion about the mean on the given qubits.
    Standard construction: H^n X^n (multi-controlled Z) X^n H^n
    """
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)


    qc.h(qubits[-1])
    if n == 1:
        qc.z(qubits[0])
    elif n == 2:
        qc.cx(qubits[0], qubits[1])
    else:
        qc.mcx(qubits[:-1], qubits[-1])  # multi-controlled X
    qc.h(qubits[-1])

    qc.x(qubits)
    qc.h(qubits)



# Oracle: phase flip on |MARKED>

def phase_oracle_marked_state(qc: QuantumCircuit, qubits: List[int], marked: str) -> None:
    """
    Flip the phase of the computational basis state |marked>.
    marked is a bitstring like "10" or "101" (len == len(qubits)).
    Convention here: marked[0] corresponds to qubits[0] (left-to-right).
    """
    if len(marked) != len(qubits) or any(b not in "01" for b in marked):
        raise ValueError("marked must be a bitstring of length n containing only 0/1")

    # Map |marked> -> |11..1> with X on qubits where marked bit is 0
    for i, bit in enumerate(marked):
        if bit == "0":
            qc.x(qubits[i])

    # Apply phase flip on |11..1> using multi-controlled Z
    qc.h(qubits[-1])
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cx(qubits[0], qubits[1])
    else:
        qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])

    # Uncompute mapping
    for i, bit in enumerate(marked):
        if bit == "0":
            qc.x(qubits[i])


# Build Grover circuit

def grover_circuit(n: int, marked: str, iters: int | None = None) -> QuantumCircuit:
    """
    Build an n-qubit Grover circuit that searches for the single marked state.
    If iters is None, uses the typical choice floor(pi/4 * sqrt(2^n)).
    """
    if iters is None:
        iters = max(1, int(math.floor((math.pi / 4.0) * math.sqrt(2**n))))

    q = QuantumRegister(n, "q")
    c = ClassicalRegister(n, "c")
    qc = QuantumCircuit(q, c)

    # Start in uniform superposition
    qc.h(q)

    # Grover iterations: oracle + diffusion
    for _ in range(iters):
        phase_oracle_marked_state(qc, list(range(n)), marked)
        diffusion(qc, list(range(n)))

    qc.measure(q, c)
    qc.name = f"Grover n={n}, marked={marked}, iters={iters}"
    return qc



# Run on BasicSimulator

def run_counts(qc: QuantumCircuit, shots: int = 2000, seed: int = 1) -> dict:
    backend = BasicSimulator()
    tqc = transpile(qc, backend, optimization_level=1)
    result = backend.run(tqc, shots=shots, seed_simulator=seed).result()
    return result.get_counts()


def pretty_top_counts(counts: dict, top_k: int = 6) -> List[Tuple[str, int]]:
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


if __name__ == "__main__":

    # Example 1: 2 qubits

    n2 = 2
    MARKED_2 = "10"         
    qc2 = grover_circuit(n2, MARKED_2, iters=1)  
    counts2 = run_counts(qc2, shots=2000, seed=7)

    print("\n", qc2.name)
    print(qc2.draw())
    print("Top counts:", pretty_top_counts(counts2))
    print("All counts:", counts2)


    # Example 2: 3 qubits

    n3 = 3
    MARKED_3 = "101"
    
    qc3 = grover_circuit(n3, MARKED_3, iters=None)
    counts3 = run_counts(qc3, shots=4000, seed=7)

    print("\n", qc3.name)
    print(qc3.draw())
    print("Top counts:", pretty_top_counts(counts3))
    print("All counts:", counts3)

    p2 = counts2.get(MARKED_2, 0) / 2000
    p3 = counts3.get(MARKED_3, 0) / 4000
    print(f"\nSuccess probability (n=2): P({MARKED_2}) = {p2:.3f}")
    print(f"Success probability (n=3): P({MARKED_3}) = {p3:.3f}")