"""Fast exact kissing checks for equal-norm integer vectors."""

from __future__ import annotations

import numpy as np


def verify_integer_equal_norm(pts: np.ndarray) -> dict:
    """pts: (m, n) integer array, all same squared length N.

    Kissing iff 2 <vi,vj> <= N for i!=j, i.e. max off-diagonal inner product <= N/2.
    """
    pts = np.asarray(pts)
    if pts.ndim != 2:
        raise ValueError("expected 2d array")
    m, n = pts.shape
    P = pts.astype(np.int32, copy=False)
    n2 = np.sum(P * P, axis=1)
    if n2.min() != n2.max():
        return {
            "ok": False,
            "reason": f"unequal norms {int(n2.min())}..{int(n2.max())}",
            "count": m,
            "dim": n,
        }
    N = int(n2[0])
    if N % 2 != 0:
        # bound is N/2; if N odd, 2 ip <= N means ip <= floor(N/2)
        bound = N // 2
    else:
        bound = N // 2
    G = P @ P.T
    iu = np.triu_indices(m, k=1)
    off = G[iu]
    mx = int(off.max()) if off.size else None
    n_viol = int(np.sum(off > bound))
    n_tight = int(np.sum(off == bound))
    # distinct
    uniq = np.unique(P, axis=0)
    ok = n_viol == 0 and uniq.shape[0] == m and N > 0
    return {
        "ok": bool(ok),
        "count": m,
        "dim": n,
        "norm2": N,
        "bound_unnormalized": bound,
        "max_offdiag_unnormalized": mx,
        "n_tight_pairs": n_tight,
        "n_violations": n_viol,
        "distinct": bool(uniq.shape[0] == m),
        "max_offdiag_unit": f"{mx}/{N}" if mx is not None else None,
    }
