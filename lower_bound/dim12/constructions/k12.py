"""Coxeter–Todd lattice K12 via the hexacode (Conway–Sloane 2-base).

Complex lattice A6^(2) = { z in Z[ω]^6 : σ(z) in the hexacode },
then K12 is the real form in R^12. Minimal vectors have real norm^2 = 4;
there are 756 of them (the lattice kissing number).

Citations:
* Conway–Sloane, Math. Proc. Camb. Phil. Soc. 93 (1983), 421–440
  https://doi.org/10.1017/S0305004100060746
* Nebe–Sloane Catalogue: https://www.math.rwth-aachen.de/~Gabriele.Nebe/LATTICES/K12.html
* Wikipedia: https://en.wikipedia.org/wiki/Coxeter%E2%80%93Todd_lattice
"""

from __future__ import annotations

import itertools
from typing import Iterable

# F4 encoded as 0,1,w=2,w2=3 with XOR addition and the table product.
F4_MUL = (
    (0, 0, 0, 0),
    (0, 1, 2, 3),
    (0, 2, 3, 1),
    (0, 3, 1, 2),
)


def f4_mul(a: int, b: int) -> int:
    return F4_MUL[a][b]


def f4_add(a: int, b: int) -> int:
    return a ^ b


def hexacode_words() -> list[tuple[int, ...]]:
    """64 words of the [6,3,4] hexacode over F4.

    Parametric form (Conway–Sloane / SPLAG): for a,b,c in F4,
        (a, b, c, a+b+c, ω²a+ωb+c, ωa+ω²b+c).
    """
    words = []
    for a, b, c in itertools.product(range(4), repeat=3):
        s = f4_add(f4_add(a, b), c)
        t = f4_add(f4_add(f4_mul(3, a), f4_mul(2, b)), c)  # w2*a + w*b + c
        u = f4_add(f4_add(f4_mul(2, a), f4_mul(3, b)), c)  # w*a + w2*b + c
        words.append((a, b, c, s, t, u))
    assert len(set(words)) == 64
    mind = 6
    for i, u in enumerate(words):
        for v in words[i + 1 :]:
            d = sum(x != y for x, y in zip(u, v))
            mind = min(mind, d)
    assert mind == 4, mind
    return words


HEXACODE = set(hexacode_words())


def sigma(a: int, b: int) -> int:
    """Z[ω] → F4, ω ↦ w."""
    return (a & 1) | ((b & 1) << 1)


def n_eis(a: int, b: int) -> int:
    return a * a - a * b + b * b


# Eisenstein integers of norms 0,1,3,4
EIS_BY_NORM: dict[int, list[tuple[int, int]]] = {0: [(0, 0)], 1: [], 3: [], 4: []}
for a in range(-3, 4):
    for b in range(-3, 4):
        n = n_eis(a, b)
        if n in (1, 3, 4):
            EIS_BY_NORM[n].append((a, b))
assert len(EIS_BY_NORM[1]) == 6
assert len(EIS_BY_NORM[3]) == 6
assert len(EIS_BY_NORM[4]) == 6


def _real_coords(zs: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Map (a+bω) to R^2 as (a - b/2, b*√3/2) stored as Q(√3) triples (p,q,d)
    meaning (p + q*sqrt(3))/d.
    """
    out: list[tuple[int, int, int]] = []
    for a, b in zs:
        # Re = (2a - b)/2 , Im = b * √3 / 2
        out.append((2 * a - b, 0, 2))
        out.append((0, b, 2))
    return out


def minimal_vectors_eis() -> list[list[tuple[int, int]]]:
    """756 vectors in Z[ω]^6 of Eisenstein-norm sum 4 with hexacode reduction."""
    found: list[list[tuple[int, int]]] = []

    def consider(zs: list[tuple[int, int]]) -> None:
        red = tuple(sigma(a, b) for a, b in zs)
        if red in HEXACODE:
            found.append(zs)

    # (4,0^5)
    for i in range(6):
        for e in EIS_BY_NORM[4]:
            zs = [(0, 0)] * 6
            zs[i] = e
            consider(zs)

    # (1,3,0^4)
    for i, j in itertools.permutations(range(6), 2):
        for e1 in EIS_BY_NORM[1]:
            for e3 in EIS_BY_NORM[3]:
                zs = [(0, 0)] * 6
                zs[i] = e1
                zs[j] = e3
                consider(zs)

    # (1^4, 0^2)
    for supp in itertools.combinations(range(6), 4):
        for es in itertools.product(EIS_BY_NORM[1], repeat=4):
            zs = [(0, 0)] * 6
            for idx, e in zip(supp, es):
                zs[idx] = e
            consider(zs)

    # unique up to identity
    uniq = []
    seen = set()
    for zs in found:
        t = tuple(zs)
        if t not in seen:
            seen.add(t)
            uniq.append(zs)
    return uniq


def k12_minima_qsqrt3() -> list[list[tuple[int, int, int]]]:
    """Unit vectors: minima / 2, coordinates in Q(sqrt(3))."""
    mins = minimal_vectors_eis()
    if len(mins) != 756:
        raise RuntimeError(f"expected 756 minima, got {len(mins)}")
    unit = []
    for zs in mins:
        rc = _real_coords(zs)
        # divide by 2: (p + q√3)/d  -> (p + q√3)/(2d)
        unit.append([(p, q, 2 * d) for (p, q, d) in rc])
    return unit


def inner_qsqrt3(u, v) -> tuple[int, int, int]:
    from math import gcd

    p = q = 0
    den = 1
    # dens are 2 or 4 typically. Use scale 4.
    for (a1, b1, d1), (a2, b2, d2) in zip(u, v):
        s1 = 4 // d1
        s2 = 4 // d2
        A1, B1 = a1 * s1, b1 * s1
        A2, B2 = a2 * s2, b2 * s2
        p += A1 * A2 + 3 * B1 * B2
        q += A1 * B2 + A2 * B1
    # inner = (p + q√3) / 16
    g = gcd(gcd(abs(p), abs(q)), 16)
    return (p // g, q // g, 16 // g)


def le_half_qsqrt3(ip: tuple[int, int, int]) -> bool:
    a, b, d = ip
    # a + b√3 <= d/2  ⇒  2a - d + 2b √3 <= 0
    s = 2 * a - d
    t = 2 * b
    if t == 0:
        return s <= 0
    if t > 0:
        if s > 0:
            return False
        # √3 <= -s/t ⇒ 3 t^2 <= s^2
        return 3 * t * t <= s * s
    if s <= 0:
        return True
    return 3 * t * t >= s * s


def is_one_qsqrt3(ip: tuple[int, int, int]) -> bool:
    a, b, d = ip
    return b == 0 and a == d


def verify_k12(vecs) -> dict:
    n = len(vecs)
    for i, v in enumerate(vecs):
        if not is_one_qsqrt3(inner_qsqrt3(v, v)):
            return {"ok": False, "reason": f"not unit {i}", "ip": inner_qsqrt3(v, v)}
        for j in range(i + 1, n):
            ip = inner_qsqrt3(v, vecs[j])
            if not le_half_qsqrt3(ip):
                return {"ok": False, "reason": f"ip({i},{j})={ip}>1/2", "count": n}
    return {"ok": True, "count": n, "dim": 12, "max_offdiag": "<=1/2", "method": "K12_hexacode"}


def triple_to_str_qsqrt3(t: tuple[int, int, int]) -> str:
    a, b, d = t
    if b == 0:
        if a == 0:
            return "0"
        if d == 1:
            return str(a)
        return f"{a}/{d}"
    if a == 0:
        if abs(b) == 1:
            return f"{'-' if b < 0 else ''}sqrt(3)/{d}"
        return f"{b}*sqrt(3)/{d}"
    return f"({a}{b:+d}*sqrt(3))/{d}"
