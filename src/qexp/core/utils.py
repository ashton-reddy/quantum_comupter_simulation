from __future__ import annotations
import json
from typing import Any, Dict


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def circuit_features(qc) -> dict:
    ops = qc.count_ops()
    return {
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "width": int(qc.num_qubits),
        "op_counts": {str(k): int(v) for k, v in ops.items()},
    }
