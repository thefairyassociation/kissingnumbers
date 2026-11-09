#!/usr/bin/env python3
"""Leech-lattice 12-dimensional sections (coordinate and diagonal)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constructions.leech import (
    golay24_codewords,
    golay_stats,
    leech_minima_integer,
    coordinate_12_section,
    diagonal_section,
)

ROOT = Path(__file__).resolve().parent.parent


def max_off(X: np.ndarray) -> float:
    if len(X) < 2:
        return float("nan")
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    return float(G.max())


def main() -> None:
    t0 = time.time()
    C = golay24_codewords()
    st = golay_stats(C)
    print("Golay24", st, flush=True)
    A, B = leech_minima_integer()
    print("Leech type A", len(A), "type B", len(B), "total", len(A) + len(B), flush=True)

    Xc = coordinate_12_section(A, B, list(range(12)))
    Xd = diagonal_section(A, B)
    report = {
        "golay": st,
        "leech_typeA": int(len(A)),
        "leech_typeB": int(len(B)),
        "coord_section_12": int(len(Xc)),
        "coord_maxIP_float": max_off(Xc) if len(Xc) else None,
        "diagonal_section": int(len(Xd)),
        "diagonal_maxIP_float": max_off(Xd) if len(Xd) else None,
        "seconds": round(time.time() - t0, 2),
        "note": "Coordinate/diagonal sections of Leech; counts expected << 841.",
    }
    print(json.dumps(report, indent=2))
    (ROOT / "configs" / "leech_sections.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "progress.log").open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"LEECH_SECTIONS {json.dumps(report)}\n"
        )


if __name__ == "__main__":
    main()
