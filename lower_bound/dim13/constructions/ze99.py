#!/usr/bin/env python3
"""Exact Zinoviev–Ericson 1154 kissing configuration in R^13.

Vectors all have squared Euclidean norm 16. The kissing condition is
<vi,vj> <= 8 for i != j, equivalent to unit inner products <= 1/2.

Layers (all in exact Q(sqrt(3)) arithmetic):
  816  tetrads: 51 four-subsets of the first 12 coordinates, all 16 sign
        patterns of (±2,±2,±2,±2,0^9).
  288  diamonds: Steiner S(5,6,12) hexads (and a 1-factor plus complements)
        as minus-sets of (±1)^12, last coordinate ±2.
    2  axials: ±4 e_12.
   48  irrationals: for each of the first 12 axes, all four vectors with
        ±2 sqrt(3) on that axis and ±2 on coordinate 12.

This reproduces the record K(13) >= 1154 of Zinoviev–Ericson 1999; it is a
baseline, not a new bound.

References:
  V. A. Zinoviev, T. Ericson, Problems Inform. Transmission 35 (1999) 287–294.
  https://www.mathnet.ru/eng/ppi457
  Henry Cohn, kissing-number table https://cohn.mit.edu/kissing-numbers
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "ze99_structure.json"


def _signs16():
    for s in range(16):
        yield tuple(1 if (s >> k) & 1 else -1 for k in range(4))


def tetrads(supports: list[int]) -> list[list[tuple[int, int]]]:
    """Each coordinate is (a, b) meaning a + b*sqrt(3)."""
    out = []
    for mask in supports:
        idxs = [i for i in range(12) if mask & (1 << i)]
        assert len(idxs) == 4, (mask, idxs)
        for sg in _signs16():
            v = [(0, 0)] * 13
            for k, i in enumerate(idxs):
                v[i] = (2 * sg[k], 0)
            out.append(v)
    return out


def diamonds(hexads: list[list[int]], factor: list[list[int]]) -> list[list[tuple[int, int]]]:
    patterns = []
    for h in hexads:
        pat = [1] * 12
        for i in h:
            pat[i] = -1
        patterns.append(tuple(pat))
    for edge in factor:
        pat = [1] * 12
        for i in edge:
            pat[i] = -1
        patterns.append(tuple(pat))
        patterns.append(tuple(-x for x in pat))  # weight-10 minus set
    # unique (hexads already closed under complement)
    patterns = list(dict.fromkeys(patterns))
    out = []
    for pat in patterns:
        for last in (2, -2):
            v = [(pat[i], 0) for i in range(12)] + [(last, 0)]
            out.append(v)
    return out


def axials() -> list[list[tuple[int, int]]]:
    up = [(0, 0)] * 12 + [(4, 0)]
    dn = [(0, 0)] * 12 + [(-4, 0)]
    return [up, dn]


def irrationals() -> list[list[tuple[int, int]]]:
    """48 vectors: ±2 sqrt(3) e_i ± 2 e_12, i=0..11."""
    out = []
    for i in range(12):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [(0, 0)] * 13
                v[i] = (0, 2 * s1)  # 2 s1 sqrt(3)
                v[12] = (2 * s2, 0)
                out.append(v)
    return out


def generate_ab() -> list[list[tuple[int, int]]]:
    data = json.loads(DATA.read_text())
    vecs = []
    vecs.extend(tetrads(data["tetrad_supports_bitmask"]))
    vecs.extend(diamonds(data["diamond_hexads"], data["diamond_factor"]))
    vecs.extend(axials())
    vecs.extend(irrationals())
    return vecs


def ab_to_strings(v: list[tuple[int, int]]) -> list[str]:
    out = []
    for a, b in v:
        if b == 0:
            out.append(str(a))
        elif a == 0:
            if b == 1:
                out.append("sqrt(3)")
            elif b == -1:
                out.append("-sqrt(3)")
            else:
                out.append(f"{b}*sqrt(3)")
        else:
            sign = "+" if b > 0 else "-"
            bb = abs(b)
            tail = "sqrt(3)" if bb == 1 else f"{bb}*sqrt(3)"
            out.append(f"{a}{sign}{tail}" if a != 0 else f"{sign}{tail}")
    return out


def generate_string_vectors() -> list[list[str]]:
    return [ab_to_strings(v) for v in generate_ab()]


def inner_ab(u: list[tuple[int, int]], v: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (p, q) so that <u,v> = p + q sqrt(3)."""
    p = 0
    q = 0
    for (a, b), (c, d) in zip(u, v):
        p += a * c + 3 * b * d
        q += a * d + b * c
    return p, q


def leq_p_plus_q_sqrt3(p: int, q: int, bound: int) -> bool:
    """True iff p + q sqrt(3) <= bound."""
    return _nonneg_ab(bound - p, -q)


def _nonneg_ab(p: int, q: int) -> bool:
    """p + q sqrt(3) >= 0."""
    if q == 0:
        return p >= 0
    if q > 0:
        if p >= 0:
            return True
        return p * p <= 3 * q * q
    # q < 0
    if p < 0:
        return False
    return p * p >= 3 * q * q


def verify_ab(vecs: list[list[tuple[int, int]]], bound: int = 8) -> dict:
    n = len(vecs)
    d = len(vecs[0])
    for i, v in enumerate(vecs):
        p, q = inner_ab(v, v)
        if q != 0 or p != 16:
            return {"ok": False, "reason": f"vector {i} has norm^2={p}+{q}*sqrt(3) != 16"}
    keyset = {}
    for i, v in enumerate(vecs):
        k = tuple(v)
        if k in keyset:
            return {"ok": False, "reason": f"duplicate {keyset[k]} and {i}"}
        keyset[k] = i
    n_tight = 0
    worst = None
    worst_pair = None
    for i in range(n):
        ui = vecs[i]
        for j in range(i + 1, n):
            p, q = inner_ab(ui, vecs[j])
            if not leq_p_plus_q_sqrt3(p, q, bound):
                return {
                    "ok": False,
                    "reason": f"inner({i},{j})={p}+{q}*sqrt(3) > {bound}",
                    "count": n,
                }
            if q == 0 and p == bound:
                n_tight += 1
            # track max: compare p+q sqrt3
            if worst is None:
                worst = (p, q)
                worst_pair = (i, j)
            else:
                # worst < current iff current - worst > 0
                dp = p - worst[0]
                dq = q - worst[1]
                if _nonneg_ab(dp, dq) and (dp, dq) != (0, 0):
                    worst = (p, q)
                    worst_pair = (i, j)
    return {
        "ok": True,
        "count": n,
        "dim": d,
        "norm2": 16,
        "n_tight_pairs": n_tight,
        "max_offdiag_unnormalized": (
            str(worst[0]) if worst[1] == 0 else f"{worst[0]}+{worst[1]}*sqrt(3)"
        ),
        "max_offdiag_unit": "1/2" if worst == (8, 0) else None,
        "max_offdiag_pair": worst_pair,
        "distinct": True,
        "all_offdiag_leq_bound": True,
        "method": "Zinoviev-Ericson 1999 reproduction (exact Q(sqrt(3)))",
    }


if __name__ == "__main__":
    vecs = generate_ab()
    print("generated", len(vecs))
    r = verify_ab(vecs)
    print(r)
