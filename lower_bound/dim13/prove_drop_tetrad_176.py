#!/usr/bin/env python3
"""Bound for the one-tetrad-deletion layers: at most 176 points, so no record.

Context.  search_ze99_drop_tetrad_layer.py (PR #8) deletes one of ZE99's 51
tetrad supports (16 vectors, 816 -> 800) and asks whether a half-height layer
can then have 177 points, drawn from the primitive directions of magnitude
<= 2 compatible with the remaining 800 tetrads.  Two such layers would give

    800 + 2 + 2 * 177 = 1156 > 1154,

and one of 177 with one of 176 would give 1155.  If neither layer can exceed
176 the total is at most 800 + 2 + 2 * 176 = 1154: a tie, not a record.

Its CP-SAT searches ended UNKNOWN at the time limit for every one of the seven
orbit representatives.  The hard part of that model is the 4096 sign vectors
(+-1)^12, which form a binary code of length 12 and minimum distance 4;
deciding the size of such a code by search is a classical hard problem.

This script removes that part by theory and leaves only small clique problems.

Write a layer as C + R: sign vectors C, and everything else R (axes, triples
and the direction types that the deletion makes available).  Call a direction
of R a *forbidder* on a 3-set S with sign pattern sigma if it conflicts with
every sign vector x with x restricted to S equal to sigma (a whole subcube of
codimension 3).  For a layer let m be the largest number of distinct patterns
forbidden on one 3-set.  Then

  * |C| <= beta(m), where beta(0..3) = 144, 132, 120, 100: the allowed
    patterns on that 3-set split into faces of a 3-cube, and a face of
    dimension 0, 1, 2 holds at most A(9,4) = 20, A(10,4) = 40, A(11,4) = 72
    code words (beta is decreasing, so beta(m) <= 100 for m >= 3);
  * |R| <= w(m), the largest set of pairwise compatible directions of R in
    which no 3-set has more than m forbidden patterns.  w is increasing.

Hence   layer <= max( 144 + w(0), 132 + w(1), 120 + w(2), 100 + w(8) ).

w(0), w(1), w(2), w(8) are maximum-clique problems on at most 648 vertices
with a few side constraints.  They are solved here with CP-SAT; where the
solver stops at the time limit its proven upper bound is used, which is all
the argument needs.  --cross-check repeats all four with HiGHS (mixed-integer
programming, a clique-cover formulation: a different model and a different
algorithm) and derives the bound a second time from HiGHS alone.  It also
runs NetworkX's pure-Python branch-and-bound clique on w(0) and w(8) where
that finishes within --networkx-seconds.  Exact values claimed by two solvers
are asserted to agree.

Everything else is exact integer arithmetic.  The seven representatives are
confirmed to cover all 51 supports by computing the automorphism group of the
tetrad system (permutations of the 12 coordinates preserving the 51 supports).

Outside inputs: A(9,4) = 20, A(10,4) = 40, A(11,4) = 72, A(12,4) = 144
(Best, Brouwer, MacWilliams, Odlyzko, Sloane 1978), and the correctness of the
solvers' optimality / bound claims.  The second is why --cross-check exists.

usage:
    python3 lower_bound/dim13/prove_drop_tetrad_176.py [--cross-check] [--seconds 240]
        [--drops 0 1 4 5 8 11 19] [--json-out FILE]
Needs numpy and ortools; --cross-check also needs networkx and scipy.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "constructions"))
import search_ze99_drop_tetrad_layer as drop_search  # noqa: E402
from ze99 import generate_ab  # noqa: E402

TARGET = 177
BETA = {0: 144, 1: 132, 2: 120, 3: 100}      # see prove_joint_layer_168.py, which computes these


def automorphism_orbits(supports: list[frozenset[int]]) -> tuple[int, list[list[int]]]:
    """Order of the coordinate automorphism group of the support family, and the orbits on supports."""
    fam = set(supports)
    pts = range(12)
    deg = Counter(p for s in supports for p in s)
    autos = []

    def extend(perm: dict[int, int]):
        if len(perm) == 12:
            if all(frozenset(perm[p] for p in s) in fam for s in supports):
                autos.append(dict(perm))
            return
        p = len(perm)
        for q in pts:
            if q in perm.values() or deg[q] != deg[p]:
                continue
            perm[p] = q
            # every support inside the mapped points must map to a support
            done = set(perm)
            if all(frozenset(perm[x] for x in s) in fam for s in supports if s <= done):
                extend(perm)
            del perm[p]

    extend({})
    index = {s: i for i, s in enumerate(supports)}
    seen, orbits = set(), []
    for i, s in enumerate(supports):
        if i in seen:
            continue
        orb = sorted({index[frozenset(a[p] for p in s)] for a in autos})
        seen.update(orb)
        orbits.append(orb)
    return len(autos), orbits


def analyse(support_matrix: np.ndarray, drop: int):
    q, n, _ = drop_search.enumerate_candidates(support_matrix, drop)
    q, n = q.astype(np.int64), n.astype(np.int64)
    sign = np.all(np.abs(q) == 1, axis=1)
    C, R, nR = q[sign], q[~sign], n[~sign]
    assert len(C) == 4096 and len({tuple(r) for r in C}) == 4096
    # sign vectors: compatible iff Hamming distance >= 4
    dcc = C @ C.T
    off = ~np.eye(4096, dtype=bool)
    assert np.array_equal(((dcc <= 0) | (9 * dcc * dcc <= 144))[off], ((12 - dcc) // 2 >= 4)[off])
    # other directions among themselves: cos <= 1/3  <=>  dot <= 0 or 9 dot^2 <= |q|^2 |r|^2
    drr = R @ R.T
    comp = (drr <= 0) | (9 * drr * drr <= nR[:, None] * nR[None, :])
    np.fill_diagonal(comp, True)
    # a direction against a sign vector
    dcr = C @ R.T
    conflict = (dcr > 0) & (9 * dcr * dcr > 12 * nR[None, :])
    # which directions forbid a whole codimension-3 subcube
    subcubes = []
    for S in itertools.combinations(range(12), 3):
        for sig in itertools.product([1, -1], repeat=3):
            subcubes.append((S, np.all(C[:, list(S)] == np.array(sig), axis=1)))
    forb: dict[int, list[int]] = {}
    for r in range(len(R)):
        col = conflict[:, r]
        if col.sum() >= 512:
            ids = [i for i, (_, mask) in enumerate(subcubes) if col[mask].all()]
            if ids:
                forb[r] = ids
    return R, nR, comp, subcubes, forb


def by_support(subcubes, forb):
    out: dict[tuple, dict[int, list[int]]] = {}
    for r, ids in forb.items():
        for i in ids:
            out.setdefault(subcubes[i][0], {}).setdefault(i, []).append(r)
    return out


def w_cpsat(R, comp, subcubes, forb, m: int | None, seconds: float, workers: int) -> dict:
    """m = None: no side constraint.  Returns the best value found and the proven upper bound."""
    from ortools.sat.python import cp_model
    M = cp_model.CpModel()
    x = [M.NewBoolVar(f"x{i}") for i in range(len(R))]
    for i, j in np.argwhere(~comp):
        if i < j:
            M.AddBoolOr([x[i].Not(), x[j].Not()])
    if m == 0:
        for r in forb:
            M.Add(x[r] == 0)
    elif m is not None:
        for _, pats in by_support(subcubes, forb).items():
            ys = []
            for i, rs in pats.items():
                y = M.NewBoolVar(f"y{i}")
                for r in rs:
                    M.AddImplication(x[r], y)
                ys.append(y)
            M.Add(sum(ys) <= m)
    M.Maximize(sum(x))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_workers = workers
    status = solver.Solve(M)
    return {"solver": "CP-SAT", "status": solver.StatusName(status),
            "found": int(round(solver.ObjectiveValue())), "upper_bound": int(np.floor(solver.BestObjectiveBound() + 1e-6))}


def _networkx_clique(comp_sub, out):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(comp_sub)))
    for i in range(len(comp_sub)):
        for j in np.flatnonzero(comp_sub[i, i + 1:]):
            G.add_edge(i, i + 1 + int(j))
    _, weight = nx.max_weight_clique(G, weight=None)
    out.put(int(weight))


def w_networkx(R, comp, forb, exclude_forbidders: bool, seconds: float) -> int | None:
    """Exact maximum clique by NetworkX's branch and bound; None if it does not finish in time."""
    import multiprocessing as mp
    keep = np.ones(len(R), dtype=bool)
    if exclude_forbidders:
        keep[list(forb)] = False
    idx = np.flatnonzero(keep)
    ctx = mp.get_context("fork")
    out = ctx.Queue()
    proc = ctx.Process(target=_networkx_clique, args=(comp[np.ix_(idx, idx)], out))
    proc.start()
    proc.join(seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None
    return out.get() if not out.empty() else None


def w_highs(R, comp, subcubes, forb, m: int | None, seconds: float) -> dict:
    """The same problem as w_cpsat, as a mixed-integer programme (m = None: no side constraint)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    nR = len(R)
    ids = sorted({i for v in forb.values() for i in v})
    ypos = {i: nR + k for k, i in enumerate(ids)}
    rows, cols, vals, ub = [], [], [], []
    r = 0
    conflict = ~comp
    np.fill_diagonal(conflict, False)
    covered = np.zeros_like(conflict)
    for i in np.argsort(-conflict.sum(1)):               # cover every conflict edge by a conflict clique
        for j in np.flatnonzero(conflict[i] & ~covered[i]):
            group, cand = [int(i), int(j)], conflict[i] & conflict[j]
            while cand.any():
                k = int(np.flatnonzero(cand)[0])
                group.append(k)
                cand &= conflict[k]
            for a in group:
                covered[a, group] = True
                rows.append(r); cols.append(a); vals.append(1)
            ub.append(1); r += 1
    upper = np.ones(nR + len(ids))
    if m == 0:
        upper[list(forb)] = 0
    elif m is not None:
        for rr, idl in forb.items():
            for i in idl:
                rows += [r, r]; cols += [rr, ypos[i]]; vals += [1, -1]
                ub.append(0); r += 1
        groups: dict[tuple, list[int]] = {}
        for i in ids:
            groups.setdefault(subcubes[i][0], []).append(i)
        for members in groups.values():
            for i in members:
                rows.append(r); cols.append(ypos[i]); vals.append(1)
            ub.append(m); r += 1
    nv = nR + len(ids)
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    c = np.zeros(nv)
    c[:nR] = -1
    res = milp(c, constraints=LinearConstraint(A, -np.inf, np.array(ub, float)), integrality=np.ones(nv),
               bounds=Bounds(0, upper), options={"time_limit": seconds})
    dual = getattr(res, "mip_dual_bound", None)
    return {"solver": "HiGHS", "status": int(res.status),
            "found": None if res.fun is None else int(round(-res.fun)),
            "upper_bound": None if dual is None else int(np.floor(-dual + 1e-6))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drops", type=int, nargs="*", help="support indices to treat (default: one per orbit)")
    ap.add_argument("--seconds", type=float, default=240.0, help="time limit per clique problem")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cross-check", action="store_true")
    ap.add_argument("--networkx-seconds", type=float, default=300.0,
                    help="wall-clock limit for each NetworkX clique computation in --cross-check")
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()

    original = generate_ab()
    masks, support_matrix = drop_search.tetrad_supports(original)
    supports = [frozenset(int(i) for i in np.flatnonzero(row)) for row in support_matrix]
    assert len(supports) == 51 and all(len(s) == 4 for s in supports)
    order, orbits = automorphism_orbits(supports)
    reps = [orb[0] for orb in orbits]
    assert sum(len(o) for o in orbits) == 51
    print(f"tetrad system: automorphism group of order {order}; orbits on the 51 supports have sizes "
          f"{[len(o) for o in orbits]}, representatives {reps}")
    drops = a.drops if a.drops else reps
    if not a.drops:
        print("treating one support per orbit covers all 51 deletions")

    record = {"target": TARGET, "automorphism_group_order": order, "orbits": orbits, "results": []}
    worst = 0
    cross_ok = True
    for drop in drops:
        t0 = time.time()
        R, nR, comp, subcubes, forb = analyse(support_matrix, drop)
        res = {"drop": drop, "support": sorted(supports[drop]), "directions": int(len(R)),
               "norm_squared_counts": {str(k): int(v) for k, v in sorted(Counter(nR.tolist()).items())},
               "forbidders": len(forb)}
        w = {}
        for key, m in (("w0", 0), ("w1", 1), ("w2", 2), ("w_all", None)):
            w[key] = w_cpsat(R, comp, subcubes, forb, m, a.seconds, a.workers)
        terms = {"m=0": BETA[0] + w["w0"]["upper_bound"], "m=1": BETA[1] + w["w1"]["upper_bound"],
                 "m=2": BETA[2] + w["w2"]["upper_bound"], "m>=3": BETA[3] + w["w_all"]["upper_bound"]}
        bound = max(terms.values())
        res.update({"cpsat": w, "terms": terms, "layer_bound": bound, "seconds": round(time.time() - t0, 1)})
        if a.cross_check:
            x = {key: w_highs(R, comp, subcubes, forb, m, a.seconds)
                 for key, m in (("w0", 0), ("w1", 1), ("w2", 2), ("w_all", None))}
            ok = all(v["upper_bound"] is not None for v in x.values())
            xterms = {"m=0": BETA[0] + x["w0"]["upper_bound"], "m=1": BETA[1] + x["w1"]["upper_bound"],
                      "m=2": BETA[2] + x["w2"]["upper_bound"], "m>=3": BETA[3] + x["w_all"]["upper_bound"]} if ok else {}
            nxv = {"w0": w_networkx(R, comp, forb, True, a.networkx_seconds),
                   "w_all": w_networkx(R, comp, forb, False, a.networkx_seconds)}
            # where two solvers both claim an exact value they must agree
            for key in x:
                if w[key]["status"] == "OPTIMAL" and x[key]["status"] == 0:
                    assert w[key]["found"] == x[key]["found"], (drop, key, w[key], x[key])
                if key in nxv and nxv[key] is not None and w[key]["status"] == "OPTIMAL":
                    assert w[key]["found"] == nxv[key], (drop, key, w[key], nxv[key])
            res["cross_check"] = {"highs": x, "terms": xterms, "layer_bound": max(xterms.values()) if ok else None,
                                  "networkx": nxv}
        record["results"].append(res)
        worst = max(worst, bound)
        line = (f"drop {drop:2d} {sorted(supports[drop])}: {len(R)} directions;  "
                f"w(0)={w['w0']['upper_bound']} w(1)<={w['w1']['upper_bound']} w(2)<={w['w2']['upper_bound']} "
                f"w(all)={w['w_all']['upper_bound']};  layer <= {bound}")
        if a.cross_check:
            cc = res["cross_check"]
            line += (f";  HiGHS: layer <= {cc['layer_bound']};  NetworkX: w(0)={cc['networkx']['w0']} "
                     f"w(all)={cc['networkx']['w_all']}")
            if cc["layer_bound"] is None or cc["layer_bound"] >= TARGET:
                cross_ok = False
        print(line, flush=True)

    record["worst_layer_bound"] = worst
    record["proved"] = worst < TARGET
    if a.cross_check:
        record["cross_check_agrees"] = cross_ok
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(record, indent=1) + "\n")
    if worst < TARGET:
        print(f"\nPROVED: after deleting any one tetrad support, a half-height layer from this family has at most "
              f"{worst} points.\nA layer of {TARGET} would be needed to pass 1154; the most this family can give is "
              f"800 + 2 + 2*{worst} = {802 + 2 * worst}.")
        return 0
    print(f"\nNOT PROVED: the bound is {worst}, which does not exclude {TARGET}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
