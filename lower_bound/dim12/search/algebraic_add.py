#!/usr/bin/env python3
"""Search for extra exact algebraic unit vectors that fit the 840-point core.

All inner-product screening uses float64 only as a filter; any survivor is
re-checked in exact arithmetic (Q(sqrt(2)), Q(sqrt(3)), Q(sqrt(5)), Q(sqrt(6)),
or a compositum via sympy).

This script does not move the 840 points.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constructions.clebsch840 import construction_840, vectors_as_float, vectors_as_strings

ROOT = Path(__file__).resolve().parent.parent
HALF = 0.5
# Conservative float filter: anything above this is rejected.
FLOAT_TOL = 1e-10


def already_in(X: np.ndarray, v: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.max(X @ v) > 1.0 - tol)


def fits_float(X: np.ndarray, v: np.ndarray) -> float:
    return float(np.max(X @ v))


def all_signed_sparse(abs_pattern: list[float]) -> np.ndarray:
    """All distinct vectors obtained by placing abs_pattern (including zeros) with signs."""
    n = 12
    nz = [i for i, a in enumerate(abs_pattern) if a != 0]
    # We treat abs_pattern as a *multiset of absolute values* assigned to coordinates.
    # Caller should pass a length-12 list of nonnegative values.
    mag = np.array(abs_pattern, dtype=np.float64)
    positions = np.where(mag > 0)[0]
    k = len(positions)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=k)))
    out = np.zeros((len(signs), n), dtype=np.float64)
    out[:, positions] = signs * mag[positions]
    return out


def patterns_from_support(support: tuple[int, ...], value: float) -> np.ndarray:
    k = len(support)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=k)))
    out = np.zeros((len(signs), 12), dtype=np.float64)
    out[:, list(support)] = signs * value
    return out


def scan_equal_weight(X: np.ndarray, weight: int, value: float, name: str) -> list[np.ndarray]:
    hits: list[np.ndarray] = []
    n_cands = 0
    for supp in itertools.combinations(range(12), weight):
        C = patterns_from_support(supp, value)
        n_cands += len(C)
        ips = C @ X.T  # (m, 840)
        mx = ips.max(axis=1)
        for c, m in zip(C, mx):
            if m <= HALF + FLOAT_TOL and not already_in(X, c):
                hits.append(c.copy())
    print(f"  {name}: candidates={n_cands} hits={len(hits)}")
    return hits


def scan_mixed_half_root2(X: np.ndarray) -> list[np.ndarray]:
    """Vectors with one ±1/√2 and two ±1/2 (unit: 1/2 + 1/4 + 1/4 = 1)."""
    hits: list[np.ndarray] = []
    a = 2.0 ** -0.5
    n_cands = 0
    for i in range(12):
        rest = [j for j in range(12) if j != i]
        for j, k in itertools.combinations(rest, 2):
            for s in itertools.product((-1.0, 1.0), repeat=3):
                v = np.zeros(12)
                v[i] = s[0] * a
                v[j] = s[1] * 0.5
                v[k] = s[2] * 0.5
                n_cands += 1
                m = fits_float(X, v)
                if m <= HALF + FLOAT_TOL and not already_in(X, v):
                    hits.append(v.copy())
    print(f"  mixed_1/sqrt2_two_halves: candidates={n_cands} hits={len(hits)}")
    return hits


def scan_clebsch_style_other_half(X: np.ndarray) -> list[np.ndarray]:
    """48-system vectors but with coordinates split across the two R^6 factors.

    This is a structured perturbation of the equator/floor alphabet.
    """
    hits: list[np.ndarray] = []
    sqrt2 = 2.0 ** 0.5
    n_cands = 0
    # Try placing the 6D 48-system into a coordinate 6-subset that is not
    # {0..5} or {6..11}.
    from constructions.clebsch840 import sixty_block, triple_to_float

    block = [[triple_to_float(c) for c in v] for v in sixty_block()]
    block = np.array(block, dtype=np.float64)
    # some mixed 6-subsets
    subsets = [
        (0, 1, 2, 6, 7, 8),
        (0, 1, 2, 3, 6, 7),
        (0, 2, 4, 6, 8, 10),
        (0, 1, 2, 3, 4, 6),
        (1, 2, 3, 7, 8, 9),
        (0, 1, 6, 7, 8, 9),
        tuple(range(1, 7)),
        (0, 1, 2, 3, 8, 9),
    ]
    for subset in subsets:
        for b in block:
            v = np.zeros(12)
            for t, idx in enumerate(subset):
                v[idx] = b[t]
            n_cands += 1
            m = fits_float(X, v)
            if m <= HALF + FLOAT_TOL and not already_in(X, v):
                hits.append(v.copy())
    print(f"  sixty_on_other_6subsets: candidates={n_cands} hits={len(hits)}")
    return hits


def sympy_unit_and_ip_ok(core_strings: list[list[str]], extra: list[str]) -> bool:
    vecs = [[sp.simplify(sp.sympify(c)) for c in row] for row in extra]
    # extra is a list of coordinate strings for one vector
    v = [sp.simplify(sp.sympify(c)) for c in extra]
    nrm = sp.simplify(sum(c * c for c in v))
    if nrm != 1:
        return False
    half = sp.Rational(1, 2)
    for row in core_strings:
        u = [sp.sympify(c) for c in row]
        ip = sp.simplify(sum(a * b for a, b in zip(u, v)))
        if ip > half:
            return False
    return True


def float_to_exact_guess(v: np.ndarray) -> list[str] | None:
    """Map a float vector onto a small exact alphabet, if possible."""
    alphabet = {
        0.0: "0",
        1.0: "1",
        -1.0: "-1",
        0.5: "1/2",
        -0.5: "-1/2",
        1 / 3: "1/3",
        -1 / 3: "-1/3",
        (2.0 ** 0.5) / 3: "sqrt(2)/3",
        -(2.0 ** 0.5) / 3: "-sqrt(2)/3",
        (2.0 ** 0.5) / 4: "sqrt(2)/4",
        -(2.0 ** 0.5) / 4: "-sqrt(2)/4",
        (2.0 ** -0.5): "1/sqrt(2)",
        -(2.0 ** -0.5): "-1/sqrt(2)",
        3 ** -0.5: "1/sqrt(3)",
        -(3 ** -0.5): "-1/sqrt(3)",
        5 ** -0.5: "1/sqrt(5)",
        -(5 ** -0.5): "-1/sqrt(5)",
        6 ** -0.5: "1/sqrt(6)",
        -(6 ** -0.5): "-1/sqrt(6)",
        8 ** -0.5: "1/sqrt(8)",
        -(8 ** -0.5): "-1/sqrt(8)",
        12 ** -0.5: "1/sqrt(12)",
        -(12 ** -0.5): "-1/sqrt(12)",
        2 / 3: "2/3",
        -2 / 3: "-2/3",
        (3 ** 0.5) / 2: "sqrt(3)/2",
        -(3 ** 0.5) / 2: "-sqrt(3)/2",
        (3 ** 0.5) / 4: "sqrt(3)/4",
        -(3 ** 0.5) / 4: "-sqrt(3)/4",
        0.25: "1/4",
        -0.25: "-1/4",
    }
    out: list[str] = []
    for x in v:
        found = None
        for val, s in alphabet.items():
            if abs(x - val) < 1e-10:
                found = s
                break
        if found is None:
            return None
        out.append(found)
    return out


def main() -> None:
    t0 = time.time()
    core = construction_840()
    X = vectors_as_float(core)
    core_s = vectors_as_strings(core)
    print("core", X.shape, "float max off", float(((X @ X.T) - np.eye(len(X))).max()))

    families = []
    families.append(("wt2_1/sqrt2", scan_equal_weight(X, 2, 2.0 ** -0.5, "wt2_1/sqrt2")))
    families.append(("wt3_1/sqrt3", scan_equal_weight(X, 3, 3.0 ** -0.5, "wt3_1/sqrt3")))
    families.append(("wt4_1/2", scan_equal_weight(X, 4, 0.5, "wt4_1/2")))
    families.append(("wt5_1/sqrt5", scan_equal_weight(X, 5, 5.0 ** -0.5, "wt5_1/sqrt5")))
    families.append(("wt6_1/sqrt6", scan_equal_weight(X, 6, 6.0 ** -0.5, "wt6_1/sqrt6")))
    families.append(("wt8_1/sqrt8", scan_equal_weight(X, 8, 8.0 ** -0.5, "wt8_1/sqrt8")))
    families.append(("wt12_1/sqrt12", scan_equal_weight(X, 12, 12.0 ** -0.5, "wt12_1/sqrt12")))
    families.append(("mixed_half_root2", scan_mixed_half_root2(X)))
    families.append(("sixty_reembedded", scan_clebsch_style_other_half(X)))

    # Cross-check hits against each other: largest subset with mutual IP <= 1/2.
    all_hits: list[tuple[str, np.ndarray]] = []
    for name, hits in families:
        for h in hits:
            all_hits.append((name, h))
    print("TOTAL FLOAT HITS", len(all_hits))

    exact_hits = []
    for name, h in all_hits:
        s = float_to_exact_guess(h)
        if s is None:
            print("  skip (no exact alphabet match)", name, np.round(h, 5))
            continue
        ok = sympy_unit_and_ip_ok(core_s, s)
        print("  exact", name, ok, s)
        if ok:
            exact_hits.append((name, s))

    # mutual compatibility
    kept: list[tuple[str, list[str]]] = []
    parsed_kept: list[list[sp.Expr]] = []
    half = sp.Rational(1, 2)
    for name, s in exact_hits:
        v = [sp.sympify(c) for c in s]
        good = True
        for u in parsed_kept:
            ip = sp.simplify(sum(a * b for a, b in zip(u, v)))
            if ip > half:
                good = False
                break
        if good:
            kept.append((name, s))
            parsed_kept.append(v)

    report = {
        "core": 840,
        "float_hits": len(all_hits),
        "exact_hits_vs_core": len(exact_hits),
        "mutually_compatible_extras": len(kept),
        "total_if_added": 840 + len(kept),
        "extras": [{"method": n, "vector": s} for n, s in kept],
        "seconds": round(time.time() - t0, 3),
    }
    out = ROOT / "configs" / "algebraic_add_to_840.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    log = ROOT / "progress.log"
    with log.open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"ALGEBRAIC_ADD core=840 exact_hits={len(exact_hits)} "
            f"compatible_extras={len(kept)} total={840+len(kept)}\n"
        )


if __name__ == "__main__":
    main()
