#!/usr/bin/env python3
"""Three 24-cell blocks (8 orts + 16 cube vertices each) plus 2+2 bridges.

Unlike D4 roots (coordinates 1/√2), cube vertices have |coord|=1/2 and
saturate but do not violate the bridge pairing constraint |v_i|+|v_j|<=1.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constructions.clebsch840 import cmp_le_half, is_one, inner_qsqrt2, triple_to_sympy_str

ROOT = Path(__file__).resolve().parent.parent


def block24(offset: int) -> list[list[tuple[int, int, int]]]:
    z = (0, 0, 1)
    vecs = []
    # 8 orts
    for i in range(4):
        for s in (1, -1):
            v = [z] * 12
            v[offset + i] = (s, 0, 1)
            vecs.append(v)
    # 16 cube vertices (±1/2)^4
    for signs in itertools.product((-1, 1), repeat=4):
        v = [z] * 12
        for i, s in enumerate(signs):
            v[offset + i] = (s, 0, 2)
        vecs.append(v)
    return vecs


def all_2_2_bridges(off_a: int, off_b: int, same_color: bool) -> list[list[tuple[int, int, int]]]:
    z = (0, 0, 1)
    K4 = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]
    if same_color:
        pairs = [(e1, e2) for fac in K4 for e1 in fac for e2 in fac]
    else:
        edges = list(itertools.combinations(range(4), 2))
        pairs = [(e1, e2) for e1 in edges for e2 in edges]
    vecs = []
    for e1, e2 in pairs:
        for signs in itertools.product((-1, 1), repeat=4):
            v = [z] * 12
            v[off_a + e1[0]] = (signs[0], 0, 2)
            v[off_a + e1[1]] = (signs[1], 0, 2)
            v[off_b + e2[0]] = (signs[2], 0, 2)
            v[off_b + e2[1]] = (signs[3], 0, 2)
            vecs.append(v)
    return vecs


def triples_to_float(vecs):
    s2 = 2.0**0.5
    return np.array([[(a + b * s2) / d for a, b, d in v] for v in vecs], dtype=np.float64)


def unique_triples(vecs):
    seen = set()
    out = []
    for v in vecs:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            out.append(v)
    return out


def greedy_exact(core, cands, tol=1e-12):
    X = triples_to_float(core)
    kept = list(core)
    for v in cands:
        fv = np.array([(a + b * (2.0**0.5)) / d for a, b, d in v])
        mx = float(np.max(X @ fv))
        if mx <= 0.5 + 1e-10 and mx < 1 - 1e-8:
            # exact check vs kept
            ok = True
            for u in kept:
                ip = inner_qsqrt2(u, v)
                if not cmp_le_half(ip):
                    ok = False
                    break
            if not is_one(inner_qsqrt2(v, v)):
                ok = False
            if ok:
                kept.append(v)
                X = np.vstack([X, fv])
    return kept


def verify_all(vecs) -> dict:
    n = len(vecs)
    max_off = None
    for i in range(n):
        if not is_one(inner_qsqrt2(vecs[i], vecs[i])):
            return {"ok": False, "reason": f"not unit {i}", "count": n}
        for j in range(i + 1, n):
            ip = inner_qsqrt2(vecs[i], vecs[j])
            if not cmp_le_half(ip):
                return {"ok": False, "reason": f"ip {i},{j}={ip}", "count": n}
            if max_off is None or (ip[0] / ip[2] > max_off[0] / max_off[2] and ip[1] == 0 and max_off[1] == 0):
                max_off = ip
    return {"ok": True, "count": n, "max_offdiag": triple_to_sympy_str(max_off) if max_off else None}


def main() -> None:
    t0 = time.time()
    core = unique_triples(block24(0) + block24(4) + block24(8))
    print("core 24*3", len(core), flush=True)
    assert len(core) == 72

    sc = unique_triples(
        all_2_2_bridges(0, 4, True)
        + all_2_2_bridges(0, 8, True)
        + all_2_2_bridges(4, 8, True)
    )
    print("same-color bridges", len(sc), flush=True)
    kept_sc = greedy_exact(core, sc)
    print("same-color total", len(kept_sc), flush=True)

    allb = unique_triples(
        all_2_2_bridges(0, 4, False)
        + all_2_2_bridges(0, 8, False)
        + all_2_2_bridges(4, 8, False)
    )
    print("all 2+2 bridges", len(allb), flush=True)
    kept_all = greedy_exact(core, allb)
    print("all 2+2 total", len(kept_all), flush=True)

    best = kept_all if len(kept_all) >= len(kept_sc) else kept_sc
    result = verify_all(best)
    print("exact", result, flush=True)

    report = {
        "core": 72,
        "same_color_count": len(kept_sc),
        "all_2plus2_count": len(kept_all),
        "best": result,
        "seconds": round(time.time() - t0, 2),
    }
    (ROOT / "configs" / "three_24cells.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
    with (ROOT / "progress.log").open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"THREE_24CELLS same_color={len(kept_sc)} all_2plus2={len(kept_all)} "
            f"exact_ok={result.get('ok')} count={result.get('count')}\n"
        )

    if result.get("ok") and result["count"] > 840:
        # dump full config
        from constructions.clebsch840 import vectors_as_strings

        payload = {
            "dim": 12,
            "count": result["count"],
            "method": "three_24cells_plus_2plus2_bridges",
            "max_offdiag": result.get("max_offdiag"),
            "vectors": [[triple_to_sympy_str(c) for c in v] for v in best],
        }
        p = ROOT / "configs" / f"three_24cells_{result['count']}.json"
        p.write_text(json.dumps(payload, separators=(",", ":")))
        print("wrote", p)


if __name__ == "__main__":
    main()
