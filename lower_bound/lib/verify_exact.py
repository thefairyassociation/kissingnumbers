"""Exact-arithmetic verifier for kissing configurations.

A configuration is stored as JSON:
  {"dimension": n, "count": N, "scale2": s,        # squared norm of the raw vectors
   "field": "Q" | "Q(sqrt(D))",
   "vectors": [[c0, c1, ...], ...]}
Each coordinate is either an integer/rational string (field Q) or a pair
[a, b] meaning a + b*sqrt(D)  (field Q(sqrt(D))), with a, b rationals.

Check: Gram diagonal == scale2 exactly, off-diagonal <= scale2/2 exactly,
vectors distinct, count > record.
"""
import json, re, sys
from fractions import Fraction

RECORDS = {12: 841, 13: 1154, 14: 1932}

def _f(x, D=0):
    """Coordinate -> (a, b) with value a + b*sqrt(D).

    Accepts [a, b] pairs, plain rationals, and sympy-style strings such as
    "2*sqrt(3)", "-1/2", "3 - 2*sqrt(3)"."""
    if isinstance(x, list):
        return (Fraction(str(x[0])), Fraction(str(x[1])))
    s = str(x).replace(" ", "")
    try:
        return (Fraction(s), Fraction(0))
    except ValueError:
        pass
    import sympy
    e = sympy.nsimplify(sympy.sympify(s))
    r = sympy.sqrt(D)
    a = sympy.simplify(e.subs(r, 0))
    b = sympy.simplify(sympy.expand((e - a) / r))
    assert a.is_rational and b.is_rational, f"cannot parse coordinate {x!r} in Q(sqrt({D}))"
    return (Fraction(str(a)), Fraction(str(b)))

def load(path):
    d = json.load(open(path))
    fld = d.get("field")
    if fld is None:                                  # sniff sqrt(D) out of the coordinates
        ds = {int(m) for v in d["vectors"] for c in v
              for m in re.findall(r"sqrt\((\d+)\)", str(c))}
        assert len(ds) <= 1, f"multiple radicals {ds}: not a single quadratic field"
        D = ds.pop() if ds else 0
        fld = "Q" if D == 0 else f"Q(sqrt({D}))"
    else:
        D = 0 if fld == "Q" else int(re.search(r"sqrt\((\d+)\)", fld).group(1))
    d["field"] = fld
    V = [[_f(c, D) for c in v] for v in d["vectors"]]
    return d, V, D

def dot(u, v, D):
    """Exact inner product in Q(sqrt(D)); returns (a,b) meaning a + b*sqrt(D)."""
    a = Fraction(0); b = Fraction(0)
    for (p, q), (r, s) in zip(u, v):
        a += p * r + q * s * D
        b += p * s + q * r
    return (a, b)

def le(x, c):
    """Exact test  a + b*sqrt(D) <= c  with c rational, D >= 0."""
    a, b, D = x[0], x[1], x[2]
    d = c - a                      # need b*sqrt(D) <= d
    if b == 0:
        return d >= 0
    if b > 0:
        return d >= 0 and b * b * D <= d * d
    return d >= 0 or b * b * D >= d * d

def verify(path, record=None, verbose=True):
    d, V, D = load(path)
    n, N = d.get("dimension", len(V[0])), len(V)
    s2 = Fraction(str(d["scale2"])) if "scale2" in d else dot(V[0], V[0], D)[0]
    assert all(len(v) == n for v in V), "dimension mismatch"
    if d.get("count") is not None:
        assert d["count"] == N, f'count field {d["count"]} != {N} vectors'
    seen = set(); 
    for v in V:
        k = tuple((str(a), str(b)) for a, b in v)
        assert k not in seen, "duplicate vector"
        seen.add(k)
    half = s2 / 2
    worst = None; ties = 0
    for i in range(N):
        g = dot(V[i], V[i], D)
        assert g == (s2, Fraction(0)), f"vector {i} has norm^2 {g}, expected {s2}"
        for j in range(i + 1, N):
            g = dot(V[i], V[j], D)
            assert le((g[0], g[1], D), half), f"pair ({i},{j}) inner product {g} > {half}"
            if g == (half, Fraction(0)):
                ties += 1
            if worst is None or (g[0], g[1]) > worst:
                worst = (g[0], g[1])
    rec = RECORDS.get(n) if record is None else record
    res = {"ok": True, "dimension": n, "count": N, "scale2": str(s2),
           "field": d.get("field", "Q"), "distinct": True,
           "max_offdiag_raw": f"{worst[0]}+{worst[1]}*sqrt({D})" if worst[1] else str(worst[0]),
           "max_offdiag_unit": str(Fraction(worst[0], 1) / s2) if worst[1] == 0 else "irrational",
           "tight_pairs": ties, "record": rec, "beats_record": N > rec,
           "method": d.get("method", "")}
    if verbose:
        print(json.dumps(res, indent=2))
    return res

if __name__ == "__main__":
    verify(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)


def verify_integer(path, record=None, verbose=True, cross_check=20000):
    """Fast exact path for all-integer coordinates.

    numpy int64 products are EXACT (not floating point) provided no overflow;
    we prove the bound |<u,v>| <= n * max|c|^2 < 2^63 before using them, and
    independently recompute a random sample of entries with Python big ints.
    """
    import json, random
    import numpy as np
    d = json.load(open(path))
    rows = d["vectors"]
    d.setdefault("dimension", len(rows[0]))
    A = np.array([[int(str(c)) for c in v] for v in rows], dtype=np.int64)
    N, n = A.shape
    mx = int(np.abs(A).max())
    bound = n * mx * mx
    assert bound < 2 ** 62, f"int64 overflow risk: bound {bound}"
    s2 = int(d["scale2"]) if "scale2" in d else int((A[0] * A[0]).sum())
    assert (A * A).sum(1).min() == s2 and (A * A).sum(1).max() == s2, "norms differ"
    G = A @ A.T
    np.fill_diagonal(G, -(10 ** 9))
    worst = int(G.max())
    half2 = s2                                   # compare 2*<u,v> against s2
    assert 2 * worst <= half2, f"max off-diagonal {worst} exceeds {s2}/2"
    assert len({tuple(r) for r in A.tolist()}) == N, "duplicate vectors"
    rng = random.Random(12345)
    for _ in range(cross_check):                 # independent big-int check
        i = rng.randrange(N); j = rng.randrange(N)
        if i == j: continue
        g = sum(int(a) * int(b) for a, b in zip(rows[i], rows[j]))
        assert 2 * g <= s2, f"pair ({i},{j}) inner product {g}"
        assert g == int(G[i, j]), "int64 / big-int mismatch"
    rec = RECORDS.get(n) if record is None else record
    res = {"ok": True, "dimension": n, "count": N, "scale2": s2, "field": "Q (integer)",
           "distinct": True, "max_offdiag_raw": worst,
           "max_offdiag_unit": str(Fraction(worst, s2)),
           "tight_pairs": int((G == s2 // 2).sum() // 2) if s2 % 2 == 0 else None,
           "record": rec, "beats_record": N > rec,
           "overflow_bound": bound, "bigint_cross_checks": cross_check,
           "method": d.get("method", "")}
    if verbose:
        print(json.dumps(res, indent=2))
    return res
