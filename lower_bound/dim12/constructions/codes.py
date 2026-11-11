"""Binary/ternary code spherical codes in R^12."""

from __future__ import annotations

import itertools
from typing import Iterable


def extended_ternary_golay_generator() -> list[list[int]]:
    """[12,6,6] extended ternary Golay over F3={0,1,2}.

    Built from the quadratic-residue [11,6,5] code (QR mod 11: 1,3,4,5,9)
    plus an overall-parity coordinate so that the sum of coords is 0 in F3.
    """
    qr = {1, 3, 4, 5, 9}
    # circulant generator from QR (including 0) of length 11, 5 independent cyclic
    # shifts plus the all-ones, then extend.
    base = [1 if (i in qr or i == 0) else 0 for i in range(11)]
    rows = []
    for s in range(5):
        rows.append(base[s:] + base[:s])
    rows.append([1] * 11)
    # extend: x12 = -sum_{1..11} so total sum 0
    ext = []
    for r in rows:
        s = sum(r) % 3
        ext.append(r + [(3 - s) % 3])
    return ext


def ternary_linear_code(gen: list[list[int]]) -> list[tuple[int, ...]]:
    k = len(gen)
    n = len(gen[0])
    words = []
    for coeffs in itertools.product(range(3), repeat=k):
        w = [0] * n
        for c, row in zip(coeffs, gen):
            if c == 0:
                continue
            for j, a in enumerate(row):
                w[j] = (w[j] + c * a) % 3
        words.append(tuple(w))
    return words


def f3_to_pm(c: int) -> int:
    return 0 if c == 0 else (1 if c == 1 else -1)


def golay_weight6_unit_strings() -> list[list[str]]:
    """Weight-6 codewords as ±1/√6 spherical vectors (and skip 0)."""
    words = ternary_linear_code(extended_ternary_golay_generator())
    out = []
    seen = set()
    for w in words:
        coords = tuple(f3_to_pm(c) for c in w)
        wt = sum(abs(x) for x in coords)
        if wt != 6:
            continue
        if coords in seen or tuple(-x for x in coords) in seen:
            # keep both ± as distinct sphere points
            pass
        seen.add(coords)
        vec = [("0" if x == 0 else ("1/sqrt(6)" if x > 0 else "-1/sqrt(6)")) for x in coords]
        out.append(vec)
    return out
