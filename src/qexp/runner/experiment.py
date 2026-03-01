from __future__ import annotations

from typing import Any, Dict

from qiskit import transpile
from qiskit_aer import AerSimulator

from qexp.core.noise import NoiseParams, build_noise_model
from qexp.core.utils import circuit_features

from qexp.algorithms.grover import GroverConfig, build_grover_circuit, grover_metrics
from qexp.algorithms.qaoa_max_cut import QAOAConfig, build_qaoa_maxcut_circuit, qaoa_metrics


def run_one_experiment(spec: Dict[str, Any]) -> Dict[str, Any]:
    algo = spec["algorithm"]
    seed = int(spec.get("seed", 0))
    noise_cfg = spec.get("noise")

    noise_params = NoiseParams(**noise_cfg) if noise_cfg else NoiseParams()
    noise_model = build_noise_model(noise_params) if noise_cfg else None

    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)

    if algo == "grover":
        cfg = GroverConfig(**spec["algo_cfg"])
        qc = build_grover_circuit(cfg)
        tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=seed)
        counts = sim.run(tqc, shots=cfg.shots).result().get_counts()
        result = grover_metrics(counts, cfg)

        return {
            "algorithm": algo,
            "seed": seed,
            "noise": noise_cfg,
            "noise_params_effective": noise_params.__dict__,
            "algo_cfg": cfg.__dict__,
            "circuit": circuit_features(tqc),
            "result": result,
        }

    if algo == "qaoa_maxcut":
        cfg = QAOAConfig(**spec["algo_cfg"])
        qc = build_qaoa_maxcut_circuit(cfg)
        tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=seed)
        counts = sim.run(tqc, shots=cfg.shots).result().get_counts()
        result = qaoa_metrics(counts, cfg)

        return {
            "algorithm": algo,
            "seed": seed,
            "noise": noise_cfg,
            "noise_params_effective": noise_params.__dict__,
            "algo_cfg": cfg.__dict__,
            "circuit": circuit_features(tqc),
            "result": result,
        }

    raise ValueError(f"Unknown algorithm: {algo}")