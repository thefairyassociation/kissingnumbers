#!/usr/bin/env python3
"""Dump the exact 840-point configuration and write best.json / a config file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constructions.clebsch840 import (
    construction_840,
    vectors_as_strings,
    verify_qsqrt2,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    vecs = construction_840()
    result = verify_qsqrt2(vecs)
    if not result.get("ok"):
        raise SystemExit(f"840 failed exact verification: {result}")
    payload = {
        "dim": 12,
        "count": 840,
        "method": "clebsch_48system_K6_bridges",
        "max_offdiag": "1/2",
        "field": "Q(sqrt(2))",
        "live_record_lower": 841,
        "note": "Baseline Leech–Sloane 840, exact. Reproducing 841 is not progress.",
        "citations": [
            "https://arxiv.org/abs/2606.18984",
            "https://doi.org/10.4153/CJM-1971-081-3",
            "https://cohn.mit.edu/kissing-numbers",
        ],
        "vectors": vectors_as_strings(vecs),
    }
    cfg = ROOT / "configs" / "clebsch840.json"
    cfg.write_text(json.dumps(payload, indent=None, separators=(",", ":")))
    best = {
        "dim": 12,
        "count": 840,
        "method": payload["method"],
        "max_offdiag": "1/2",
        "verified": "exact_qsqrt2",
        "config": str(cfg.relative_to(ROOT)),
        "live_record_lower": 841,
        "beaten": False,
    }
    (ROOT / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("wrote", cfg, "bytes", cfg.stat().st_size)


if __name__ == "__main__":
    main()
