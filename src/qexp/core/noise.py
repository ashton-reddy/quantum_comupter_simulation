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

# --------- Logging helpers + drift ---------

from dataclasses import asdict
from typing import Any, Dict
import numpy as np

def noise_to_log_dict(p: "NoiseParams") -> Dict[str, Any]:
    d = asdict(p)
    d["gate_fidelity_proxy_1q"] = 1.0 - p.p1
    d["gate_fidelity_proxy_2q"] = 1.0 - p.p2
    d["readout_fidelity_proxy"] = 1.0 - p.pm
    return d


@dataclass
class DriftModel:
    """
    Simple drift: base + slope*t + random-walk
    """
    base: "NoiseParams"

    slope_p1: float = 0.0
    slope_p2: float = 0.0
    slope_pm: float = 0.0
    slope_T1: float = 0.0
    slope_T2: float = 0.0

    rw_sigma_p2: float = 0.0
    rw_sigma_pm: float = 0.0

    slope_freq_shift_hz: float = 0.0
    rw_sigma_freq_shift_hz: float = 0.0

    def params_at(self, t: int, rng: np.random.Generator, state: Dict[str, float]) -> "NoiseParams":
        def rw(key: str, sigma: float) -> float:
            if sigma <= 0:
                return state.get(key, 0.0)
            state[key] = state.get(key, 0.0) + float(rng.normal(0.0, sigma))
            return state[key]

        p1 = self.base.p1 + self.slope_p1 * t
        p2 = self.base.p2 + self.slope_p2 * t + rw("rw_p2", self.rw_sigma_p2)
        pm = self.base.pm + self.slope_pm * t + rw("rw_pm", self.rw_sigma_pm)

        T1 = None if self.base.T1 is None else (self.base.T1 + self.slope_T1 * t)
        T2 = None if self.base.T2 is None else (self.base.T2 + self.slope_T2 * t)

        freq = self.base.freq_shift_hz + self.slope_freq_shift_hz * t + rw("rw_freq", self.rw_sigma_freq_shift_hz)

        # clamp probabilities
        p1 = float(np.clip(p1, 0.0, 1.0))
        p2 = float(np.clip(p2, 0.0, 1.0))
        pm = float(np.clip(pm, 0.0, 1.0))

        return NoiseParams(
            p1=p1, p2=p2, pm=pm,
            T1=T1, T2=T2,
            t1q=self.base.t1q, t2q=self.base.t2q,
            timing_jitter_s=self.base.timing_jitter_s,
            freq_shift_hz=float(freq),
            crosstalk_strength=self.base.crosstalk_strength,
        )
