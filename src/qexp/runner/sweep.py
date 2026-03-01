from __future__ import annotations

import itertools
import json
from datetime import datetime
from typing import Any, Dict

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qexp.core.noise import NoiseParams, DriftModel, build_noise_model, noise_to_log_dict
from qexp.algorithms.grover import GroverSpec, build_grover, grover_metrics
from qexp.qec.bitflip_syndrome import QECSpec, build_bitflip_syndrome_round, syndrome_metrics


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def circuit_features(qc) -> dict:
    ops = qc.count_ops()
    return {
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "num_qubits": int(qc.num_qubits),
        "op_counts": {str(k): int(v) for k, v in ops.items()},
    }


def run_counts(qc, shots: int, seed: int, noise_model):
    backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=seed)
    res = backend.run(tqc, shots=shots).result()
    return res.get_counts(), tqc


def main() -> None:
    out_path = "dataset.jsonl"

    base_seed = 7
    rng = np.random.default_rng(base_seed)

    base = NoiseParams(
        p1=0.001,
        p2=0.01,
        pm=0.02,
        T1=80e-6,
        T2=60e-6,
        timing_jitter_s=2e-9,
        freq_shift_hz=0.0,
        crosstalk_strength=0.0,
    )

    drift = DriftModel(
        base=base,
        slope_p2=2e-5,
        rw_sigma_p2=2e-4,
        slope_pm=1e-5,
        rw_sigma_pm=1e-4,
        slope_freq_shift_hz=5.0,
        rw_sigma_freq_shift_hz=20.0,
    )

    drift_state: Dict[str, float] = {}

    # -------------------------
    # Grover sweep
    # -------------------------
    grover_ns = [3, 4]
    grover_iters = [1, 2, 3, 4]
    grover_shots = 2000
    marked_by_n = {3: "101", 4: "1010"}

    run_index = 0
    for n, k in itertools.product(grover_ns, grover_iters):
        noise_params = drift.params_at(run_index, rng, drift_state)
        noise_model = build_noise_model(noise_params)

        spec = GroverSpec(n=n, marked=marked_by_n[n], iters=k, shots=grover_shots)
        qc = build_grover(spec)

        counts, tqc = run_counts(qc, grover_shots, seed=base_seed + run_index, noise_model=noise_model)
        metrics = grover_metrics(counts, spec)

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "domain": "algorithm_benchmark",
            "algorithm": "grover",
            "run_index": run_index,
            "spec": {"n": n, "marked": spec.marked, "iters": k, "shots": grover_shots},
            "noise": noise_to_log_dict(noise_params),
            "circuit": circuit_features(tqc),
            "counts_top": sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10],
            "metrics": metrics,
        }
        append_jsonl(out_path, row)
        run_index += 1

    # -------------------------
    # QEC sweep (syndrome cycles)
    # -------------------------
    qec = QECSpec(shots=4000, cycles=15, seed=base_seed)
    qc_round = build_bitflip_syndrome_round()

    prev_cycle_err = None

    for cycle in range(qec.cycles):
        noise_params = drift.params_at(run_index, rng, drift_state)
        noise_model = build_noise_model(noise_params)

        counts, tqc = run_counts(qc_round, qec.shots, seed=base_seed + run_index, noise_model=noise_model)
        metrics = syndrome_metrics(counts)

        cycle_err = metrics["error_rate_this_cycle"]
        temporal_corr = None
        if prev_cycle_err is not None:
            temporal_corr = float(cycle_err - prev_cycle_err)
        prev_cycle_err = cycle_err

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "domain": "quantum_error_correction",
            "experiment": "bitflip_code_syndrome",
            "run_index": run_index,
            "cycle": cycle,
            "noise": noise_to_log_dict(noise_params),
            "circuit": circuit_features(tqc),
            "counts": counts,
            "metrics": metrics,
            "derived": {"temporal_error_delta": temporal_corr},
        }
        append_jsonl(out_path, row)
        run_index += 1

    print(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    main()