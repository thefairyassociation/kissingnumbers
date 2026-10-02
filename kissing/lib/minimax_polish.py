#!/usr/bin/env python3
"""Local minimax polish by hinge continuation.

Repeatedly set t = (current max cosine) - delta and minimise the hinge energy
sum (g_ij - t)_+^2 with hingepol; keep the result only if the true maximum
went down, and shrink delta when it did not.  Only pairs within delta of the
maximum feel the hinge, so each subproblem is a small, local correction.  The
loop estimates the bottom of the candidate's minimax basin -- which is what
decides whether a Riesz endpoint at, say, 0.5004 hides a configuration below
1/2.  (A single hinge solve at t = 1/2 does not answer that: it trades many
small violations for a few large ones.)

usage:
    python3 kissing/lib/minimax_polish.py IN.txt OUT.txt [--delta 1e-4]
        [--min-delta 1e-9] [--iters 4000] [--rounds 200]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HINGEPOL = HERE / "hingepol"


def unit(path) -> np.ndarray:
    X = np.loadtxt(path, comments="#", ndmin=2)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def max_cos(X: np.ndarray) -> float:
    G = X @ X.T
    np.fill_diagonal(G, -2)
    return float(G.max())


def polish(X: np.ndarray, delta: float = 1e-4, min_delta: float = 1e-9, iters: int = 4000,
           rounds: int = 200, verbose: bool = True) -> np.ndarray:
    if not HINGEPOL.exists():
        subprocess.run(["make", "-s", "-C", str(HERE), "hingepol"], check=True)
    N, n = X.shape
    m = max_cos(X)
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.txt", Path(td) / "out.txt"
        for r in range(rounds):
            if delta < min_delta:
                break
            t = m - delta
            np.savetxt(fin, X, fmt="%.17g")
            out = subprocess.run([str(HINGEPOL), str(n), str(N), str(fin), str(fout),
                                  "--t", repr(t), "--iters", str(iters), "--quiet",
                                  "--margin", repr(max(0.02, 4 * delta))],
                                 capture_output=True, text=True, check=True).stdout
            Y = unit(fout)
            mY = max_cos(Y)
            if mY < m:
                gain = m - mY
                X, m = Y, mY
                delta = min(delta * 1.5, 1e-2) if gain > 0.5 * delta else delta
            else:
                delta *= 0.3
            if verbose:
                E = out.split()[0]
                print(f"round {r:3d} t={t:.12f} {E} max={mY:.12f} best={m:.12f} delta={delta:.2e}",
                      flush=True)
    return X


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--delta", type=float, default=1e-4)
    ap.add_argument("--min-delta", type=float, default=1e-9)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    X = unit(a.infile)
    m0 = max_cos(X)
    Y = polish(X, a.delta, a.min_delta, a.iters, a.rounds, not a.quiet)
    m1 = max_cos(Y)
    np.savetxt(a.outfile, Y, fmt="%.17g",
               header=f"n={Y.shape[1]} N={Y.shape[0]} max_inner={m1:.17g} minimax_polish_from={m0:.17g}")
    print(f"{a.infile}: {m0:.15f} -> {m1:.15f}  (independent recomputation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
