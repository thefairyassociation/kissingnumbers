#!/usr/bin/env python3
"""Can a dimension-12 kissing configuration serve as the equator of a layered
1155 in dimension 13?  Numerical; not a proof.

ZE99 is  2 poles + 816 equator points + 2 layers of 168 at height 1/2.  A
natural variation is to put a *better* dimension-12 configuration on the
equator (the classical 840, or the 841 found by campaign c03) and ask what
layers fit above it.  A layer point (sqrt(3)/2 u, 1/2), u a unit vector in
R^12, is compatible with an equator point (v, 0) iff  u.v <= 1/sqrt(3).

So the first question is whether *any* such u exists.  P = {x : x.v <= 1/sqrt(3)
for all v} is a polytope containing 0; a unit u exists iff P has a point of
norm >= 1.  The norm is maximised over P from many random starts with the
fixed-point iteration of ../lib/holes.py (a local method: the values printed
are lower bounds for the true maximum, so a value below 1 is evidence, not
proof, that no layer point exists).

usage:  python3 lower_bound/dim13/equator_room.py [--starts 400]
Needs numpy, scipy, sympy (to rebuild the starting configurations).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lower_bound" / "lib"))
from holes import max_norm_over_P  # noqa: E402


def load(kind: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "init.txt"
        subprocess.run([str(ROOT / "lower_bound/lib/campaign_init.sh"), kind, str(out)], cwd=ROOT, check=True,
                       capture_output=True)
        return np.loadtxt(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starts", type=int, default=400)
    a = ap.parse_args()
    ze = load("d13-1154")
    equators = {
        "ZE99 equator (816 tetrad vectors)": ze[np.abs(ze[:, 12]) < 1e-9][:, :12],
        "classical 840": load("d12-840"),
        "841 found by campaign c03": load("d12-841-found"),
    }
    half = 1 / np.sqrt(3)
    for name, E in equators.items():
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        norm, _ = max_norm_over_P(E, half=half, starts=a.starts, seed=1)
        verdict = "layer points exist" if norm >= 1 - 1e-9 else "no layer point found"
        print(f"{name:36s} |E| = {len(E):4d}   largest norm in P = {norm:.6f}   "
              f"(smallest max inner product {half / norm:.6f}; need <= {half:.6f})   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
