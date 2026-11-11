"""Exact Leech–Sloane / Clebsch 840-point kissing arrangement in R^12.

Structure (Takhanov–Assylbekov–Yun, arXiv:2606.18984, reproducing the 1971
Leech–Sloane lower bound of 840):

* 60 vectors supported on R^6 x {0}
* 60 vectors supported on {0} x R^6
* 720 bridge vectors from the unique 1-factorization of K6

Each 60-block is 12 signed coordinate vectors plus the 48-system of Example 1
in arXiv:2606.18984 (skewed Clebsch equator in R^5 together with analytic
±1/2 floors). All coordinates lie in Q(sqrt(2)).

References:
* J. Leech and N. J. A. Sloane, Sphere packings and error-correcting codes,
  Canad. J. Math. 23 (1971), 718–745.
  https://doi.org/10.4153/CJM-1971-081-3
* R. Takhanov, Z. Assylbekov, S. Yun, arXiv:2606.18984
  https://arxiv.org/abs/2606.18984
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Sequence

# 1-factorization of K6 on vertices {0,1,2,3,4,5}. Unique up to isomorphism.
K6_ONE_FACTORIZATION: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 1), (2, 3), (4, 5)),
    ((0, 2), (1, 4), (3, 5)),
    ((0, 3), (1, 5), (2, 4)),
    ((0, 4), (1, 3), (2, 5)),
    ((0, 5), (1, 2), (3, 4)),
)


def _even_sign_patterns() -> list[tuple[int, int, int, int, int]]:
    """16 sign 5-tuples with product +1 (vertices of the Clebsch graph)."""
    out: list[tuple[int, int, int, int, int]] = []
    for signs in itertools.product((-1, 1), repeat=5):
        prod = signs[0] * signs[1] * signs[2] * signs[3] * signs[4]
        if prod == 1:
            out.append(signs)  # type: ignore[arg-type]
    assert len(out) == 16
    return out


def forty_eight_system_qsqrt2() -> list[list[tuple[int, int, int]]]:
    """48-system in R^6 with exact coordinates (a + b*sqrt(2)) / d.

    Each coordinate is stored as a triple (a, b, d) meaning (a + b*sqrt(2))/d
    with d > 0.
    """
    vecs: list[list[tuple[int, int, int]]] = []
    for e1, e2, e3, e4, e5 in _even_sign_patterns():
        # equator x_ε = (√2/3 ε1, √2/3 ε2, √2/3 ε3, √2/3 ε4, 1/3 ε5)
        x = [
            (0, e1, 3),
            (0, e2, 3),
            (0, e3, 3),
            (0, e4, 3),
            (e5, 0, 3),
        ]
        # floor b_ε = -(√2/4 ε1, √2/4 ε2, √2/4 ε3, √2/4 ε4, 1/2 ε5)
        b = [
            (0, -e1, 4),
            (0, -e2, 4),
            (0, -e3, 4),
            (0, -e4, 4),
            (-e5, 0, 2),
        ]
        zero = (0, 0, 1)
        half = (1, 0, 2)
        nhalf = (-1, 0, 2)
        vecs.append([zero, *x])
        vecs.append([half, *b])
        vecs.append([nhalf, *b])
    assert len(vecs) == 48
    return vecs


def signed_orts_6() -> list[list[tuple[int, int, int]]]:
    vecs: list[list[tuple[int, int, int]]] = []
    z = (0, 0, 1)
    for i in range(6):
        for s in (1, -1):
            v = [z] * 6
            v[i] = (s, 0, 1)
            vecs.append(v)
    return vecs


def sixty_block() -> list[list[tuple[int, int, int]]]:
    return forty_eight_system_qsqrt2() + signed_orts_6()


def _pad12_first(v6: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    z = (0, 0, 1)
    return list(v6) + [z] * 6


def _pad12_second(v6: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    z = (0, 0, 1)
    return [z] * 6 + list(v6)


def bridge_vectors() -> list[list[tuple[int, int, int]]]:
    """720 bridges: same-color K6 edges, four coordinates ±1/2."""
    vecs: list[list[tuple[int, int, int]]] = []
    z = (0, 0, 1)
    for factor in K6_ONE_FACTORIZATION:
        for i, j in factor:
            for k, ell in factor:
                for signs in itertools.product((-1, 1), repeat=4):
                    v = [z] * 12
                    v[i] = (signs[0], 0, 2)
                    v[j] = (signs[1], 0, 2)
                    v[6 + k] = (signs[2], 0, 2)
                    v[6 + ell] = (signs[3], 0, 2)
                    vecs.append(v)
    assert len(vecs) == 720
    return vecs


def construction_840() -> list[list[tuple[int, int, int]]]:
    """Return 840 exact unit vectors as (a, b, d) triples in Q(sqrt(2))."""
    block = sixty_block()
    vecs = [_pad12_first(v) for v in block]
    vecs += [_pad12_second(v) for v in block]
    vecs += bridge_vectors()
    assert len(vecs) == 840
    return vecs


def triple_to_sympy_str(t: tuple[int, int, int]) -> str:
    a, b, d = t
    if b == 0:
        if a == 0:
            return "0"
        if d == 1:
            return str(a)
        return f"{a}/{d}" if a > 0 else f"-{abs(a)}/{d}"
    # b * sqrt(2) / d, optionally plus a/d
    if a == 0:
        if b == 1 and d == 1:
            return "sqrt(2)"
        if b == -1 and d == 1:
            return "-sqrt(2)"
        if abs(b) == 1:
            return f"{'-' if b < 0 else ''}sqrt(2)/{d}"
        return f"{b}*sqrt(2)/{d}"
    return f"({a}{b:+d}*sqrt(2))/{d}"


def vectors_as_strings(vecs: list[list[tuple[int, int, int]]]) -> list[list[str]]:
    return [[triple_to_sympy_str(c) for c in v] for v in vecs]


def triple_to_float(t: tuple[int, int, int]) -> float:
    a, b, d = t
    return (a + b * (2.0**0.5)) / d


def vectors_as_float(vecs: list[list[tuple[int, int, int]]]):
    import numpy as np

    return np.array([[triple_to_float(c) for c in v] for v in vecs], dtype=np.float64)


def _reduce(a: int, b: int, d: int) -> tuple[int, int, int]:
    from math import gcd

    g = gcd(gcd(abs(a), abs(b)), d)
    if g > 1:
        a //= g
        b //= g
        d //= g
    if d < 0:
        a, b, d = -a, -b, -d
    return a, b, d


def inner_qsqrt2(
    u: Sequence[tuple[int, int, int]], v: Sequence[tuple[int, int, int]]
) -> tuple[int, int, int]:
    """Inner product as reduced (a, b, d) meaning (a + b*sqrt(2))/d."""
    # Scale each coordinate by 12 so numerators are in Z[sqrt(2)].
    # 12/d in {12,6,4,3} for d in {1,2,3,4}.
    pa = pb = 0
    qa = qb = 0  # unused; we accumulate p + q√2 = 144 * inner
    p = 0
    q = 0
    for (a1, b1, d1), (a2, b2, d2) in zip(u, v):
        # u = (a1 + b1√2)/d1, scaled U = 12/d1 * (a1 + b1√2)
        s1 = 12 // d1
        s2 = 12 // d2
        A1, B1 = a1 * s1, b1 * s1
        A2, B2 = a2 * s2, b2 * s2
        p += A1 * A2 + 2 * B1 * B2
        q += A1 * B2 + A2 * B1
    # inner = (p + q √2) / 144
    return _reduce(p, q, 144)


def cmp_le_half(ip: tuple[int, int, int]) -> bool:
    """Return True iff (a + b√2)/d <= 1/2."""
    a, b, d = ip
    # s + t√2 <= 0 with s = 2a - d, t = 2b.
    s = 2 * a - d
    t = 2 * b
    if t == 0:
        return s <= 0
    if t > 0:
        # need s <= 0 and 2 t^2 <= s^2
        if s > 0:
            return False
        return 2 * t * t <= s * s
    # t < 0: if s <= 0 then both terms nonpositive; if s > 0 need |t|√2 >= s
    if s <= 0:
        return True
    return 2 * t * t >= s * s


def is_one(ip: tuple[int, int, int]) -> bool:
    a, b, d = ip
    # (a + b√2)/d == 1 iff a - d + b√2 == 0 iff b==0 and a==d
    return b == 0 and a == d


def verify_qsqrt2(
    vecs: list[list[tuple[int, int, int]]], ip_bound: tuple[int, int, int] = (1, 0, 2)
) -> dict:
    """Exact verification over Q(sqrt(2))."""
    n = len(vecs)
    dim = len(vecs[0])
    max_off = None  # store as (a,b,d) the maximum off-diagonal
    n_pairs = 0
    for i in range(n):
        ipii = inner_qsqrt2(vecs[i], vecs[i])
        if not is_one(ipii):
            return {
                "ok": False,
                "reason": f"vector {i} is not unit: {ipii}",
                "count": n,
                "dim": dim,
            }
        for j in range(i + 1, n):
            n_pairs += 1
            ip = inner_qsqrt2(vecs[i], vecs[j])
            if not cmp_le_half(ip):
                return {
                    "ok": False,
                    "reason": f"inner product ({i},{j}) = {ip} > 1/2",
                    "count": n,
                    "dim": dim,
                }
            if i != j:
                if max_off is None or _greater_qsqrt2(ip, max_off):
                    max_off = ip
    # distinctness: unit vectors with inner product 1 would fail ≤ 1/2 unless equal;
    # inner product = 1 is > 1/2, so distinctness is implied if kissing holds.
    return {
        "ok": True,
        "count": n,
        "dim": dim,
        "pairs": n_pairs,
        "max_offdiag": max_off,
        "max_offdiag_str": None if max_off is None else triple_to_sympy_str(max_off),
    }


def _greater_qsqrt2(u: tuple[int, int, int], v: tuple[int, int, int]) -> bool:
    """True iff u > v in Q(sqrt(2))."""
    a, b, d = u
    c, e, f = v
    # (a+b√2)/d - (c+e√2)/f = ((a f - c d) + (b f - e d)√2) / (d f)
    s = a * f - c * d
    t = b * f - e * d
    den = d * f
    # sign of (s + t√2)/den ; den>0
    if t == 0:
        return s > 0
    if t > 0:
        if s >= 0:
            return True
        # s < 0: s + t√2 > 0 iff 2 t^2 > s^2
        return 2 * t * t > s * s
    # t < 0
    if s <= 0:
        return False
    return 2 * t * t < s * s
