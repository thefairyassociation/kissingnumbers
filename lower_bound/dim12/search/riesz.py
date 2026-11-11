#!/usr/bin/env python3
"""Spherical Riesz-energy / hinge optimizer for kissing in R^12.

Floats are used only for search. A configuration is a success only after
exact (rational-shell or sympy) certification.

Continuation: increase s in logsumexp(s * inner_products) while also applying
a hinge on max(0, Gij - 1/2 + margin).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constructions.clebsch840 import construction_840, vectors_as_float

ROOT = Path(__file__).resolve().parent.parent
HALF = 0.5


def row_normalize(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-15)
    return X / nrm


def max_offdiag(X: np.ndarray) -> float:
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    return float(G.max())


def logsumexp_ip_loss_and_grad(X: np.ndarray, s: float) -> tuple[float, np.ndarray]:
    """L = log sum_{i!=j} exp(s Gij), G = X X^T. Rows of X are unit."""
    n = X.shape[0]
    G = X @ X.T
    off = G.copy()
    np.fill_diagonal(off, -np.inf)
    m = off.max()
    E = np.exp(s * (off - m))
    np.fill_diagonal(E, 0.0)
    Z = float(E.sum())
    loss = m + np.log(Z + 1e-300)
    # dL/dG = E / Z, diag 0
    A = E / (Z + 1e-300)
    # dL/dX = (A + A.T) X  but A already includes both (i,j) and (j,i)
    grad = (A + A.T) @ X
    # sphere projection of gradient: remove radial component
    grad = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    return float(loss), grad


def hinge_grad(X: np.ndarray, margin: float) -> tuple[float, np.ndarray]:
    """Sum of squares of max(0, Gij - (1/2 - margin)) over i < j."""
    n = X.shape[0]
    G = X @ X.T
    thr = HALF - margin
    excess = G - thr
    np.fill_diagonal(excess, 0.0)
    mask = excess > 0
    loss = 0.5 * float((excess[mask] ** 2).sum())
    A = np.zeros_like(G)
    A[mask] = excess[mask]
    grad = (A + A.T) @ X
    grad = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    return loss, grad


def adam_sphere(
    X: np.ndarray,
    steps: int,
    s_schedule: list[float],
    lr: float = 0.02,
    hinge_weight: float = 5.0,
    margin: float = 1e-4,
    seed: int = 0,
    report_every: int = 200,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = row_normalize(X)
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    steps_per = max(1, steps // max(len(s_schedule), 1))
    t = 0
    best = X.copy()
    best_mx = max_offdiag(X)
    for si, s in enumerate(s_schedule):
        for _ in range(steps_per):
            t += 1
            loss_r, g_r = logsumexp_ip_loss_and_grad(X, s)
            loss_h, g_h = hinge_grad(X, margin)
            g = g_r + hinge_weight * g_h
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            mhat = m / (1 - b1**t)
            vhat = v / (1 - b2**t)
            X = X - lr * mhat / (np.sqrt(vhat) + eps)
            X = row_normalize(X)
            if t % report_every == 0 or t == 1:
                mx = max_offdiag(X)
                if mx < best_mx:
                    best_mx = mx
                    best = X.copy()
                print(
                    f"  step={t} s={s:.1f} maxIP={mx:.9f} best={best_mx:.9f} "
                    f"riesz={loss_r:.4f} hinge={loss_h:.4e}",
                    flush=True,
                )
        # shrink lr a bit each stage
        lr *= 0.85
    mx = max_offdiag(X)
    if mx < best_mx:
        best = X.copy()
        best_mx = mx
    print(f"  done maxIP={best_mx:.12f}", flush=True)
    return best


def init_from_840_plus(k: int, seed: int, mode: str) -> np.ndarray:
    X0 = vectors_as_float(construction_840())
    rng = np.random.default_rng(seed)
    extras = []
    if mode == "signs":
        # (1/√12)(±1)^12 with extra random sign patterns, mutually far
        used = []
        while len(extras) < k:
            s = rng.choice([-1.0, 1.0], size=12)
            v = s / np.sqrt(12.0)
            if used and max(abs(v @ u) for u in used) > 0.8:
                continue
            extras.append(v)
            used.append(v)
    elif mode == "random":
        for _ in range(k):
            v = rng.normal(size=12)
            extras.append(v / np.linalg.norm(v))
    elif mode == "hole":
        # a few steps of subgradient against the 840 core
        for _ in range(k):
            x = rng.normal(size=12)
            x /= np.linalg.norm(x)
            for _it in range(400):
                ips = X0 @ x
                i = int(np.argmax(ips))
                x = x - 0.05 * X0[i]
                x /= np.linalg.norm(x)
            extras.append(x)
    else:
        raise ValueError(mode)
    return row_normalize(np.vstack([X0, np.stack(extras)]))


def snap_to_rationals(X: np.ndarray, denoms: list[int]) -> tuple[np.ndarray | None, int, float]:
    """Try rounding unnormalized integer vectors, keep those with exact unit
    inner-product inequality 4 <vi,vj>^2 <= |vi|^2 |vj|^2.
    Returns float unit vectors of the snapped config (for inspection) plus denom.
    """
    best = None
    best_d = 0
    best_mx = 1.0
    for d in denoms:
        Y = np.rint(X * d).astype(np.int64)
        # drop all-zero rows
        if np.any(np.all(Y == 0, axis=1)):
            continue
        # exact pairwise via integers
        G = Y @ Y.T  # integer
        nrms = np.sum(Y * Y, axis=1)
        n = len(Y)
        ok = True
        mx_unit = -1.0
        for i in range(n):
            if nrms[i] == 0:
                ok = False
                break
            for j in range(i + 1, n):
                ip = int(G[i, j])
                # unit IP = ip / sqrt(ni nj) <= 1/2 iff 2 ip <= sqrt(ni nj) when ip>0
                if ip > 0:
                    if 4 * ip * ip > nrms[i] * nrms[j]:
                        ok = False
                        break
                # track float unit ip
                uip = ip / np.sqrt(nrms[i] * nrms[j])
                if uip > mx_unit:
                    mx_unit = uip
            if not ok:
                break
        if ok:
            unit = Y.astype(np.float64) / np.sqrt(nrms)[:, None]
            return unit, d, float(mx_unit)
    return None, 0, best_mx


def save_numeric(X: np.ndarray, path: Path, meta: dict) -> None:
    payload = {
        **meta,
        "count": int(X.shape[0]),
        "dim": 12,
        "max_offdiag_float": max_offdiag(X),
        "vectors_float": X.tolist(),
        "note": "FLOAT SEARCH ONLY — not a certificate. Exact coords required to claim a record.",
    }
    path.write_text(json.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--mode", default="signs", choices=["signs", "random", "hole"])
    ap.add_argument("--lr", type=float, default=0.03)
    args = ap.parse_args()

    n = 840 + args.extra
    print(f"init n={n} seed={args.seed} mode={args.mode}", flush=True)
    X = init_from_840_plus(args.extra, args.seed, args.mode)
    print("init maxIP", max_offdiag(X), flush=True)
    s_schedule = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    t0 = time.time()
    X = adam_sphere(X, steps=args.steps, s_schedule=s_schedule, lr=args.lr, seed=args.seed)
    mx = max_offdiag(X)
    elapsed = time.time() - t0
    out = ROOT / "configs" / f"search_{n}_seed{args.seed}_{args.mode}.json"
    save_numeric(
        X,
        out,
        {
            "method": f"riesz_adam_{args.mode}",
            "seed": args.seed,
            "steps": args.steps,
            "seconds": round(elapsed, 2),
            "certified": False,
        },
    )
    print("wrote", out, "maxIP", mx, "elapsed", elapsed)

    # try rational snap if we have slack
    if mx < HALF:
        snapped, d, smx = snap_to_rationals(X, [2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
        print("snap", None if snapped is None else (d, smx))
        if snapped is not None and smx <= HALF:
            print("EXACT RATIONAL SNAP SUCCESS", d, smx, "count", len(snapped))

    log = ROOT / "progress.log"
    with log.open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"RIESZ n={n} seed={args.seed} mode={args.mode} maxIP_float={mx:.12f} "
            f"steps={args.steps} sec={elapsed:.1f}\n"
        )


if __name__ == "__main__":
    main()
