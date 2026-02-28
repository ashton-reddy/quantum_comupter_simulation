from __future__ import annotations
from dataclasses import dataclass
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


@dataclass(frozen=True)
class NoiseParams:
    p1: float = 0.0   # depolarizing after 1q gates
    p2: float = 0.0   # depolarizing after 2q gates
    pm: float = 0.0   # readout bit-flip probability


def build_noise_model(params: NoiseParams) -> NoiseModel:
    """
    Baseline noise:
      - Depolarizing on 1q gates with prob p1
      - Depolarizing on 2q gates with prob p2
      - Symmetric readout bit-flip with prob pm
    """
    nm = NoiseModel()

    # Quantum errors
    if params.p1 > 0:
        err1 = depolarizing_error(params.p1, 1)
        # apply to common 1q basis gates
        nm.add_all_qubit_quantum_error(err1, ["x", "y", "z", "h", "rx", "ry", "rz", "sx", "id"])

    if params.p2 > 0:
        err2 = depolarizing_error(params.p2, 2)
        # apply to common 2q basis gates
        nm.add_all_qubit_quantum_error(err2, ["cx", "cz", "swap", "rzz"])

    # Readout error (independent on each measured qubit)
    if params.pm > 0:
        ro = ReadoutError([[1 - params.pm, params.pm],
                           [params.pm, 1 - params.pm]])
        nm.add_all_qubit_readout_error(ro)

    return nm
