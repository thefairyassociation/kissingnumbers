#!/usr/bin/env python3
"""Try to add D12/D13 integer roots and other small shells to ZE99."""

from __future__ import annotations

import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "constructions"))
sys.path.insert(0, str(ROOT))

from ze99 import generate_ab, inner_ab, leq_p_plus_q_sqrt3, verify_ab, ab_to_strings  # noqa: E402
from io_config import log_progress, save_config, maybe_update_best  # noqa: E402


def to_int_and_s3(vecs):
    """Split into integer arrays a,b of shape (m,13)."""
    a = np.array([[p for p, q in v] for v in vecs], dtype=np.int16)
    b = np.array([[q for p, q in v] for v in vecs], dtype=np.int16)
    return a, b


def ip_ab_mat(a1, b1, a2, b2):
    """Return (p,q) arrays of shape (m1,m2): inner = p + q sqrt(3)."""
    p = a1.astype(np.int32) @ a2.astype(np.int32).T + 3 * (b1.astype(np.int32) @ b2.astype(np.int32).T)
    q = a1.astype(np.int32) @ b2.astype(np.int32).T + b1.astype(np.int32) @ a2.astype(np.int32).T
    return p, q


def compatible(p, q, bound=8):
    """Boolean matrix: p + q sqrt(3) <= bound."""
    # (bound-p) + (-q) sqrt(3) >= 0
    bp = bound - p
    bq = -q
    ok = np.zeros(p.shape, dtype=bool)
    # q_expr = bq
    # case bq == 0: bp >= 0
    z = bq == 0
    ok[z] = bp[z] >= 0
    pos = bq > 0
    # if bq>0: bp>=0 or bp^2 <= 3 bq^2
    ok[pos] = (bp[pos] >= 0) | ((bp[pos] < 0) & (bp[pos].astype(np.int64) ** 2 <= 3 * bq[pos].astype(np.int64) ** 2))
    neg = bq < 0
    # if bq<0: bp>=0 and bp^2 >= 3 bq^2
    ok[neg] = (bp[neg] >= 0) & (bp[neg].astype(np.int64) ** 2 >= 3 * bq[neg].astype(np.int64) ** 2)
    return ok


def d_roots(dim, include_last=True):
    """All (±2,±2,0^{dim-2}) in R^dim as (a,b=0) vectors of length 13 (pad)."""
    out = []
    last = dim if include_last else dim - 1
    for i, j in combinations(range(last), 2):
        for si, sj in product((-2, 2), repeat=2):
            v = [(0, 0)] * 13
            v[i] = (si, 0)
            v[j] = (sj, 0)
            out.append(v)
    return out


def plus4_axials(n=12):
    out = []
    for i in range(n):
        for s in (4, -4):
            v = [(0, 0)] * 13
            v[i] = (s, 0)
            out.append(v)
    return out


def sqrt5_like_irrationals():
    """Not Q(sqrt(3)); skip here."""
    return []


def try_add(base, pool, name):
    a0, b0 = to_int_and_s3(base)
    # drop duplicates of base
    base_keys = set(tuple(v) for v in base)
    cand = [v for v in pool if tuple(v) not in base_keys]
    if not cand:
        log_progress(name, 13, len(base), "fail", "empty candidate pool after dedup")
        return base, 0
    a1, b1 = to_int_and_s3(cand)
    p, q = ip_ab_mat(a1, b1, a0, b0)
    ok_vs_base = compatible(p, q).all(axis=1)
    surv = [v for v, o in zip(cand, ok_vs_base) if o]
    print(f"  {name}: {ok_vs_base.sum()} / {len(cand)} compatible with base {len(base)}")
    if not surv:
        log_progress(name, 13, len(base), "fail", "0 compatible with base")
        return base, 0
    # greedy among survivors: keep those compatible with each other
    a2, b2 = to_int_and_s3(surv)
    p2, q2 = ip_ab_mat(a2, b2, a2, b2)
    ok2 = compatible(p2, q2)
    np.fill_diagonal(ok2, True)
    # greedy by degree
    deg = ok2.sum(axis=1)
    order = np.argsort(-deg)
    kept_idx = []
    for i in order:
        if all(ok2[i, j] for j in kept_idx):
            kept_idx.append(int(i))
    added = [surv[i] for i in kept_idx]
    new = base + added
    print(f"  {name}: greedy added {len(added)} -> total {len(new)}")
    r = verify_ab(new)
    print("  verify", r.get("ok"), r.get("count"), r.get("reason", r.get("max_offdiag_unnormalized")))
    status = "pass" if r.get("ok") else "fail"
    log_progress(name, 13, len(new), status, f"added {len(added)} from pool {len(cand)}; {r}")
    if r.get("ok") and len(new) > 1154:
        path = save_config(
            dimension=13,
            count=len(new),
            vectors=[ab_to_strings(v) for v in new],
            max_off_diagonal="1/2",
            method=name,
            unit=False,
            extra={"norm2": 16, "added": len(added)},
            filename=f"record_{len(new)}_{name}.json",
            verified=True,
            verifier=r,
        )
        maybe_update_best(path)
        print("RECORD?", path)
    elif r.get("ok"):
        path = save_config(
            dimension=13,
            count=len(new),
            vectors=[ab_to_strings(v) for v in new],
            max_off_diagonal="1/2",
            method=name,
            unit=False,
            extra={"norm2": 16, "added": len(added)},
            filename=f"shell_{len(new)}_{name.replace('/','_')}.json",
            verified=True,
            verifier=r,
        )
        maybe_update_best(path)
    return new, len(added)


def main():
    ze = generate_ab()
    print("ZE99", len(ze))
    r = verify_ab(ze)
    print("baseline", r)

    # D-roots in first 12 only
    try_add(ze, d_roots(13, include_last=False), "add-D12-roots-to-ze99")
    # D-roots in all 13
    try_add(ze, d_roots(13, include_last=True), "add-D13-roots-to-ze99")
    # extra ±4 axials on first 12
    try_add(ze, plus4_axials(12), "add-axials-first12-to-ze99")

    # integer skeleton 1106 without irrationals
    integer = [v for v in ze if all(q == 0 for _, q in v)]
    print("integer skeleton", len(integer))
    try_add(integer, d_roots(13, include_last=True), "add-D13-to-1106")
    try_add(integer, plus4_axials(12), "add-axials-to-1106")
    try_add(integer, generate_ab()[-48:], "add-irrationals-to-1106")  # should recover 1154

    # remaining tetrads: all 4-subsets of 13 with all signs
    pool = []
    for comb in combinations(range(13), 4):
        for sg in product((-2, 2), repeat=4):
            v = [(0, 0)] * 13
            for k, i in enumerate(comb):
                v[i] = (sg[k], 0)
            pool.append(v)
    print("all tetrads pool", len(pool))
    try_add(ze, pool, "add-all-tetrads-to-ze99")
    try_add(integer, pool, "add-all-tetrads-to-1106")


if __name__ == "__main__":
    main()
