#!/usr/bin/env python3
"""Surgery search: iterated remove/relax/reinsert/polish on a spherical code.

The Riesz continuation (riesz.c, fastriesz.c) ends in a basin; the exact
hinge polish (hingepol.c) then decides whether that basin holds a feasible
configuration (hinge energy E = sum (g_ij - 1/2)_+^2 reaches 0) or not.  When
it does not, the leftover violation is typically concentrated on a few points.
This driver performs local surgery on that defect:

  1. pick k points, with probability proportional to their violation energy;
  2. remove them and hinge-polish the rest at t = 1/2 - slack, opening room;
  3. reinsert k points at the deepest holes (random sampling followed by a
     smooth-max refinement of the largest inner product);
  4. hinge-polish everything at t = 1/2 and accept if E went down
     (or, with --temp, by a Metropolis rule).

If E reaches 0 the configuration is polished once more at a strictly smaller
threshold, checked by an independent NumPy recomputation and written as
HIT_*.txt; certify_float.py is then run on it to produce the exact proof.

usage:
    python3 kissing/lib/surgery.py START.txt OUTDIR [--seed 0] [--kmax 12]
        [--slack 0.002] [--iters 0] [--hours 1.0] [--temp 0]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HINGEPOL = HERE / "hingepol"
CERTIFY = HERE / "certify_float.py"
FEASIBLE_E = 1e-26


def load(path) -> np.ndarray:
    X = np.loadtxt(path, comments="#", ndmin=2)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def save(path, X, note=""):
    G = X @ X.T
    np.fill_diagonal(G, -2)
    np.savetxt(path, X, fmt="%.17g",
               header=f"n={X.shape[1]} N={X.shape[0]} max_inner={G.max():.17g} {note}")


def hinge(X: np.ndarray, t: float, iters: int, work: Path, tag: str,
          fixed: int = 0) -> tuple[np.ndarray, float, float]:
    """Run hingepol; return (X, E, max)."""
    N, n = X.shape
    fin, fout = work / f"{tag}_in.txt", work / f"{tag}_out.txt"
    np.savetxt(fin, X, fmt="%.17g")
    cmd = [str(HINGEPOL), str(n), str(N), str(fin), str(fout), "--t", repr(t),
           "--iters", str(iters), "--quiet"]
    if fixed:
        cmd += ["--fixed", str(fixed)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    d = dict(kv.split("=", 1) for kv in out.split())
    return load(fout), float(d["E"]), float(d["max"])


def violation_energy(X: np.ndarray, t: float = 0.5) -> np.ndarray:
    G = X @ X.T
    np.fill_diagonal(G, -2)
    H = np.maximum(G - t, 0.0)
    return (H * H).sum(1)


def deepest_hole(Y: np.ndarray, rng: np.random.Generator, samples: int = 40000,
                 refine: int = 24) -> np.ndarray:
    """A unit vector with (locally) smallest max inner product against Y."""
    n = Y.shape[1]
    U = rng.standard_normal((samples, n))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    m = (U @ Y.T).max(1)
    cand = U[np.argsort(m)[:refine]]
    best, bu = 9.0, None
    for u in cand:
        u = u.copy()
        for beta, steps, lr in ((60.0, 150, 0.02), (400.0, 150, 0.005), (3000.0, 200, 0.001)):
            for _ in range(steps):
                g = Y @ u
                w = np.exp(beta * (g - g.max()))
                w /= w.sum()
                d = w @ Y
                d -= (d @ u) * u
                u -= lr * d
                u /= np.linalg.norm(u)
        v = (Y @ u).max()
        if v < best:
            best, bu = v, u
    return bu


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start")
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--slack", type=float, default=0.002)
    ap.add_argument("--relax-iters", type=int, default=3000)
    ap.add_argument("--polish-iters", type=int, default=6000)
    ap.add_argument("--iters", type=int, default=0, help="stop after this many moves (0: no limit)")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="Metropolis temperature on relative energy change (0: greedy)")
    a = ap.parse_args()

    if not HINGEPOL.exists():
        subprocess.run(["make", "-C", str(HERE), "hingepol"], check=True)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / f"work_s{a.seed}"
    work.mkdir(exist_ok=True)
    log = open(out / f"surgery_s{a.seed}.log", "a")
    rng = np.random.default_rng(a.seed)

    X = load(a.start)
    N, n = X.shape
    X, E, mx = hinge(X, 0.5, 20000, work, "init")
    best_E, best_X = E, X.copy()
    print(f"start {a.start}: N={N} n={n} E={E:.6e} max={mx:.12f}", flush=True)
    log.write(json.dumps({"move": 0, "E": E, "max": mx, "start": a.start}) + "\n")
    t0 = time.time()
    move = 0
    while True:
        move += 1
        if a.iters and move > a.iters:
            break
        if time.time() - t0 > 3600 * a.hours:
            break
        k = int(rng.integers(1, a.kmax + 1))
        v = violation_energy(X)
        p = v + 1e-3 * v.mean() + 1e-300
        p /= p.sum()
        drop = rng.choice(N, size=k, replace=False, p=p)
        keep = np.setdiff1d(np.arange(N), drop)
        Y, _, _ = hinge(X[keep], 0.5 - a.slack, a.relax_iters, work, "relax")
        new = []
        for _ in range(k):
            u = deepest_hole(np.vstack([Y] + [np.array(new)] if new else [Y]), rng)
            new.append(u)
        Xn = np.vstack([Y, np.array(new)])
        Xn, En, mxn = hinge(Xn, 0.5, a.polish_iters, work, "polish")
        accept = En < E
        if not accept and a.temp > 0:
            accept = rng.random() < np.exp(-(En - E) / (a.temp * E))
        rec = {"move": move, "k": k, "E_new": En, "max_new": mxn, "E": E, "accept": bool(accept),
               "t": round(time.time() - t0, 1)}
        if accept:
            X, E, mx = Xn, En, mxn
        if E < best_E:
            best_E, best_X = E, X.copy()
            save(out / f"best_s{a.seed}.txt", best_X, note=f"surgery E={best_E:.6e}")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"move {move:5d} k={k:2d} E_new={En:.6e} max_new={mxn:.9f}  "
              f"E={E:.6e} best={best_E:.6e} {'ACC' if accept else '   '} {time.time()-t0:7.1f}s",
              flush=True)
        if E <= FEASIBLE_E:
            # Strictly feasible: push below 1/2 by a margin, verify independently.
            Xf, Ef, mf = hinge(X, 0.5 - 1e-7, 50000, work, "final")
            Uf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
            G = Uf @ Uf.T
            np.fill_diagonal(G, -2)
            m_indep = float(G.max())
            hit = out / f"HIT_n{n}_N{N}_s{a.seed}.txt"
            save(hit, Uf, note=f"surgery move={move}")
            print(f"FEASIBLE: E={Ef:.3e} independent max={m_indep:.17g} -> {hit}", flush=True)
            log.write(json.dumps({"HIT": str(hit), "max": m_indep}) + "\n")
            log.flush()
            if m_indep < 0.5:
                subprocess.run([sys.executable, str(CERTIFY), str(hit),
                                "--json-out", str(hit.with_suffix(".cert.json"))])
            break
    print(f"done: best E={best_E:.6e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
