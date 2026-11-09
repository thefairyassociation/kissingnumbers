#!/usr/bin/env python3
"""Build K12 minima, verify exactly, dump config."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constructions.k12 import (
    k12_minima_qsqrt3,
    verify_k12,
    triple_to_str_qsqrt3,
    HEXACODE,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    t0 = time.time()
    print("hexacode size", len(HEXACODE), flush=True)
    vecs = k12_minima_qsqrt3()
    print("minima", len(vecs), "built in", round(time.time() - t0, 2), flush=True)
    t1 = time.time()
    result = verify_k12(vecs)
    result["seconds"] = round(time.time() - t1, 2)
    print(result, flush=True)
    if not result.get("ok"):
        raise SystemExit(2)
    payload = {
        "dim": 12,
        "count": 756,
        "method": "K12_CoxeterTodd_hexacode_2base",
        "max_offdiag": "1/2",
        "field": "Q(sqrt(3))",
        "citations": [
            "https://doi.org/10.1017/S0305004100060746",
            "https://www.math.rwth-aachen.de/~Gabriele.Nebe/LATTICES/K12.html",
            "https://en.wikipedia.org/wiki/Coxeter%E2%80%93Todd_lattice",
        ],
        "vectors": [[triple_to_str_qsqrt3(c) for c in v] for v in vecs],
    }
    path = ROOT / "configs" / "k12_minima_756.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print("wrote", path, "bytes", path.stat().st_size)
    with (ROOT / "progress.log").open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"K12_MINIMA verified count=756 exact_Qsqrt3 ok={result.get('ok')}\n"
        )


if __name__ == "__main__":
    main()
