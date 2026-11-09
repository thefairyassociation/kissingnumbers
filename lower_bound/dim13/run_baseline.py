#!/usr/bin/env python3
"""Reproduce and exact-verify the Zinoviev–Ericson 1154 baseline in dimension 13."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "constructions"))

from constructions.ze99 import generate_ab, ab_to_strings, verify_ab  # noqa: E402
from io_config import save_config, maybe_update_best, log_progress  # noqa: E402


def main() -> int:
    vecs = generate_ab()
    result = verify_ab(vecs)
    print("=== ZE99 baseline exact verification (Q(sqrt(3)) integer arithmetic) ===")
    print(json.dumps(result, indent=2))
    strings = [ab_to_strings(v) for v in vecs]
    path = save_config(
        dimension=13,
        count=len(vecs),
        vectors=strings,
        max_off_diagonal="8/16 = 1/2",
        method="Zinoviev–Ericson 1999 reproduction (tetrads + Steiner diamonds + axials + 48*sqrt(3) irrationals)",
        unit=False,
        extra={
            "norm2": "16",
            "layers": {
                "tetrads_wt4": 816,
                "diamonds": 288,
                "axials": 2,
                "irrationals_sqrt3": 48,
            },
            "baseline_not_new_bound": True,
        },
        filename="ze99_1154_exact.json",
        verified=bool(result.get("ok")),
        verifier=result,
    )
    if result.get("ok"):
        maybe_update_best(path)
        log_progress(
            "ze99-reproduction",
            13,
            len(vecs),
            "pass",
            "baseline 1154 exact; not a record improvement",
        )
        print(f"saved {path}")
        return 0
    log_progress("ze99-reproduction", 13, len(vecs), "fail", str(result))
    print("FAILED", result, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
