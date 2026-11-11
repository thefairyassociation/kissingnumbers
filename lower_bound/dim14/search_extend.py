#!/usr/bin/env python3
"""Structured searches for a dim-14 kissing configuration larger than 1932.

Families (exact Z-model, squared length 8, inner products <= 4):
  A. Extra weight-8 vectors compatible with Ganzhinov 1932.
  B. Local remove/reinsert on the weight-8 compatibility graph.
  C. Random greedy independent sets in the weight-8 graph (plus all 364 type-22).
  D. Type (2,1^4,0^9) trades against type-(2,2).
  E. Cohn–Li style all-equal vector in Q(sqrt(7)) mixed with a subset of 1932.
  F. Barnes–Wall Λ16 coordinate sections.
  G. Shortened Witt S(5,8,24) even-sign constructions.
  H. Numerical holes of the 1932 configuration (then algebraize if found).

Never claims success on floats; a beat is recorded only after verifier.py.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from constructions.ganzhinov import d14_type22, ganzhinov_1932  # noqa: E402
from verifier import verify_integer_equal_norm  # noqa: E402

LOG = ROOT / "progress.log"
BEST = ROOT / "best.json"
CFG = ROOT / "configs"


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    with LOG.open("a") as f:
        f.write(line)


def save_best(pts: np.ndarray, family: str, notes: str) -> None:
    res = verify_integer_equal_norm(pts)
    payload = {
        "n": int(pts.shape[0]),
        "dim": 14,
        "family": family,
        "notes": notes,
        "verified": res,
        "norm2": 8,
        "max_inner": 4 if res.get("ok") else None,
    }
    BEST.write_text(json.dumps(payload, indent=2))
    np.save(CFG / f"best_{pts.shape[0]}.npy", pts)
    # integer txt
    np.savetxt(CFG / f"best_{pts.shape[0]}.txt", pts, fmt="%d", delimiter=",")
    log(f"SAVED best n={pts.shape[0]} family={family} ok={res.get('ok')} beats={res.get('beats_record')}")


def is_w8(v: np.ndarray) -> bool:
    return int(np.sum(np.abs(v) == 1)) == 8 and int(np.sum(np.abs(v) == 2)) == 0


def split_types(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w8_mask = np.sum(np.abs(pts) == 1, axis=1) == 8
    return pts[w8_mask], pts[~w8_mask]


def max_inner(batch: np.ndarray, base: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """For each row of batch, max inner product with rows of base (int16)."""
    b = batch.astype(np.int16, copy=False)
    g = base.astype(np.int16, copy=False)
    out = np.empty(b.shape[0], dtype=np.int16)
    for i in range(0, b.shape[0], chunk):
        sl = b[i : i + chunk]
        dots = sl @ g.T
        out[i : i + sl.shape[0]] = dots.max(axis=1)
    return out


def generate_all_w8() -> np.ndarray:
    """All 768768 vectors in {0,±1}^14 of weight 8. Memory ~10 MB."""
    supports = list(combinations(range(14), 8))
    n = len(supports) * 256
    out = np.zeros((n, 14), dtype=np.int8)
    row = 0
    signs = np.array(list(product((-1, 1), repeat=8)), dtype=np.int8)
    for supp in supports:
        block = out[row : row + 256]
        for k, j in enumerate(supp):
            block[:, j] = signs[:, k]
        row += 256
    return out


def family_a_extra_w8(g1932: np.ndarray) -> int:
    log("FAMILY A: extra weight-8 vs frozen Ganzhinov 1932")
    w8_all = generate_all_w8()
    gset = set(map(tuple, g1932.tolist()))
    # Candidates not already in the configuration.
    # Check in batches whether max inner with g1932 is <= 4.
    # A vector already in g1932 has max inner 8 with itself; exclude those.
    mx = max_inner(w8_all, g1932, chunk=2048)
    extra = w8_all[mx <= 4]
    log(f"  weight-8 with max inner <=4 against 1932: {extra.shape[0]} (includes none of the 1568)")
    n_extra = extra.shape[0]
    if n_extra:
        # add them all if they are mutually compatible
        pts = np.concatenate([g1932, extra], axis=0)
        pts = np.unique(pts, axis=0)
        res = verify_integer_equal_norm(pts)
        log(f"  after adding extras n={pts.shape[0]} ok={res.get('ok')} viol={res.get('violations')} max={res.get('max_offdiag_inner')}")
        if res.get("ok") and pts.shape[0] > 1932:
            save_best(pts, "ganzhinov+extra_w8", "FAMILY A extras mutually ok")
        elif n_extra:
            # greedy among extras
            chosen = [g1932]
            cur = g1932
            added = 0
            order = np.random.default_rng(0).permutation(extra.shape[0])
            for i in order:
                v = extra[i : i + 1]
                if int((v.astype(np.int16) @ cur.astype(np.int16).T).max()) <= 4:
                    cur = np.concatenate([cur, v, -v], axis=0)
                    added += 2
            cur = np.unique(cur, axis=0)
            res = verify_integer_equal_norm(cur)
            log(f"  greedy extras n={cur.shape[0]} added~{added} ok={res.get('ok')}")
            if res.get("ok") and cur.shape[0] > 1932:
                save_best(cur, "ganzhinov+greedy_extra_w8", "FAMILY A")
    return n_extra


def family_b_swaps(g1932: np.ndarray, max_candidates: int = 80000) -> int:
    """Try adding a weight-8 vector by removing its conflict set if net-positive."""
    log("FAMILY B: remove/reinsert swaps on weight-8 graph")
    w8, t22 = split_types(g1932)
    w8_all = generate_all_w8()
    gset = set(map(tuple, g1932.tolist()))
    rng = np.random.default_rng(1)
    idx = rng.choice(w8_all.shape[0], size=min(max_candidates, w8_all.shape[0]), replace=False)
    cand = w8_all[idx]
    best_n = g1932.shape[0]
    best_pts = g1932
    improvements = 0
    w8_i16 = w8.astype(np.int16)
    t22_i16 = t22.astype(np.int16)
    for i in range(cand.shape[0]):
        v = cand[i]
        key = tuple(int(x) for x in v)
        if key in gset:
            continue
        dots = v.astype(np.int16) @ w8_i16.T
        conf = np.where(dots > 4)[0]
        # type-22 never conflicts with w8
        # also check -v conflicts: same indices as v because antipodal closed? 
        # If w in S and v·w > 4 then (-v)·(-w) > 4; (-v)·w = -(v·w) < -4 so |ip|>4
        # Removing conf and their antipodes.
        if conf.size == 0:
            # should have been caught in family A
            new = np.concatenate([g1932, v.reshape(1, -1), (-v).reshape(1, -1)], axis=0)
            new = np.unique(new, axis=0)
            if new.shape[0] > best_n:
                best_n = new.shape[0]
                best_pts = new
                improvements += 1
            continue
        # map conf to antipodal closure
        drop = set()
        for j in conf:
            drop.add(tuple(int(x) for x in w8[j]))
            drop.add(tuple(int(-x) for x in w8[j]))
        # also drop vectors with v·w < -4 (i.e. conflict with -v already included)
        conf_neg = np.where(dots < -4)[0]
        for j in conf_neg:
            drop.add(tuple(int(x) for x in w8[j]))
            drop.add(tuple(int(-x) for x in w8[j]))
        n_drop = len(drop)
        if n_drop >= 2:
            # net if we only add ±v: 2 - n_drop <= 0, skip unless n_drop==2 (swap, maybe unlock later)
            if n_drop > 2:
                continue
        keep_w8 = np.array([row for row in w8 if tuple(int(x) for x in row) not in drop], dtype=np.int8)
        trial = np.concatenate([t22, keep_w8, v.reshape(1, -1), (-v).reshape(1, -1)], axis=0)
        # try to reinsert some dropped? no — try to add other candidates quickly
        trial = np.unique(trial, axis=0)
        if trial.shape[0] > best_n:
            res = verify_integer_equal_norm(trial)
            if res.get("ok"):
                best_n = trial.shape[0]
                best_pts = trial
                improvements += 1
                log(f"  swap improvement n={best_n} dropped={n_drop}")
        if (i + 1) % 20000 == 0:
            log(f"  swap scanned {i+1}/{cand.shape[0]} best={best_n}")
    if best_n > 1932:
        save_best(best_pts, "swap_w8", "FAMILY B")
    log(f"  FAMILY B done best={best_n} improvements={improvements}")
    return best_n


def family_c_greedy(n_trials: int = 12, seed: int = 0) -> int:
    log(f"FAMILY C: {n_trials} random greedy independent sets in weight-8 graph")
    t22 = d14_type22()
    w8_all = generate_all_w8()
    # Work with positive hemisphere: first nonzero > 0, add antipodes at the end.
    pos_mask = []
    for v in w8_all:
        for x in v:
            if x != 0:
                pos_mask.append(x > 0)
                break
    pos = w8_all[np.array(pos_mask)]
    log(f"  positive weight-8: {pos.shape[0]}")
    best_n = 0
    best_pts = None
    rng = np.random.default_rng(seed)
    for t in range(n_trials):
        t0 = time.time()
        order = rng.permutation(pos.shape[0])
        chosen_list = []
        chosen = np.zeros((0, 14), dtype=np.int8)
        for i in order:
            v = pos[i : i + 1]
            if chosen.shape[0] == 0:
                chosen = v.copy()
                chosen_list.append(v[0])
                continue
            dmax = int((v.astype(np.int16) @ chosen.astype(np.int16).T).max())
            dmin = int((v.astype(np.int16) @ chosen.astype(np.int16).T).min())
            if dmax <= 4 and dmin >= -4:
                chosen = np.concatenate([chosen, v], axis=0)
        pts = np.concatenate([t22, chosen, -chosen], axis=0)
        pts = np.unique(pts, axis=0)
        elapsed = time.time() - t0
        log(f"  trial {t}: n={pts.shape[0]} pos={chosen.shape[0]} time={elapsed:.1f}s")
        if pts.shape[0] > best_n:
            best_n = pts.shape[0]
            best_pts = pts
            res = verify_integer_equal_norm(pts)
            log(f"    verified ok={res.get('ok')} max={res.get('max_offdiag_inner')}")
            if res.get("ok") and pts.shape[0] > 1932:
                save_best(pts, "greedy_w8", f"FAMILY C trial {t}")
                return best_n
    log(f"  FAMILY C best={best_n}")
    return best_n


def family_d_type214(g1932: np.ndarray) -> None:
    log("FAMILY D: type (2,1^4,0^9) vs type-(2,2) trades")
    # Generate all type-214
    pts = []
    for p in range(14):
        rest = [j for j in range(14) if j != p]
        for four in combinations(rest, 4):
            for s2 in (2, -2):
                for signs in product((-1, 1), repeat=4):
                    v = np.zeros(14, dtype=np.int8)
                    v[p] = s2
                    for s, j in zip(signs, four):
                        v[j] = s
                    pts.append(v)
    t214 = np.stack(pts, axis=0)
    log(f"  generated {t214.shape[0]} type-214 vectors")
    mx = max_inner(t214, g1932, chunk=2048)
    extra = t214[mx <= 4]
    log(f"  compatible with full 1932: {extra.shape[0]}")
    if extra.shape[0]:
        trial = np.unique(np.concatenate([g1932, extra], axis=0), axis=0)
        res = verify_integer_equal_norm(trial)
        log(f"  add-all n={trial.shape[0]} ok={res.get('ok')} viol={res.get('violations')}")
        if res.get("ok") and trial.shape[0] > 1932:
            save_best(trial, "ganzhinov+type214", "FAMILY D")
            return
    # Count how many type-22 each type-214 conflicts with; look for cheap trades.
    t22, w8 = split_types(g1932)[1], split_types(g1932)[0]
    # actually split_types returns w8, t22
    w8, t22 = split_types(g1932)
    t22_i = t22.astype(np.int16)
    # sample
    rng = np.random.default_rng(2)
    sample = t214[rng.choice(t214.shape[0], size=min(20000, t214.shape[0]), replace=False)]
    best_gain = -10**9
    for v in sample:
        dots22 = v.astype(np.int16) @ t22_i.T
        conf22 = np.where(np.abs(dots22) > 4)[0]
        dots8 = v.astype(np.int16) @ w8.astype(np.int16).T
        conf8 = np.where(np.abs(dots8) > 4)[0]
        n_drop = len({tuple(t22[j]) for j in conf22} | {tuple(-t22[j]) for j in conf22} | {tuple(w8[j]) for j in conf8} | {tuple(-w8[j]) for j in conf8})
        gain = 2 - n_drop  # add ±v
        if gain > best_gain:
            best_gain = gain
        if gain > 0:
            keep22 = [row for i, row in enumerate(t22) if i not in set(conf22) and tuple(-row) not in {tuple(-t22[j]) for j in conf22}]
            # simpler unique keep
            drop = {tuple(t22[j]) for j in conf22} | {tuple(-t22[j]) for j in conf22} | {tuple(w8[j]) for j in conf8} | {tuple(-w8[j]) for j in conf8}
            keep = np.array([row for row in g1932 if tuple(int(x) for x in row) not in drop], dtype=np.int8)
            trial = np.unique(np.concatenate([keep, v.reshape(1, -1), (-v).reshape(1, -1)]), axis=0)
            if trial.shape[0] > 1932:
                res = verify_integer_equal_norm(trial)
                log(f"  positive trade n={trial.shape[0]} ok={res.get('ok')}")
                if res.get("ok"):
                    save_best(trial, "type214_trade", "FAMILY D")
                    return
    log(f"  FAMILY D best single-vector gain={best_gain} (no beat)")


def family_e_allones(g1932: np.ndarray) -> None:
    log("FAMILY E: Cohn-Li all-equal vector mixed with subset of 1932")
    # Integer model uses norm2=8. All-equal unit vector is (a,...,a) with 14 a^2 = 8, a=sqrt(8/14).
    # Exact test: q^2 ip^2 <= p^2 n1 n2 with p/q=1/2.
    # For integer v of norm2=8, ip = a * sum(v_i) = sqrt(8/14)*s, cosine = s/sqrt(14*8)*sqrt(8) wait.
    # cosine = <v, u> / (||v|| ||u||) = (a s) / (sqrt(8)*1) = sqrt(8/14) s / sqrt(8) = s / sqrt(14)
    # Need |s|/sqrt(14) <= 1/2 iff |s| <= sqrt(14)/2 ≈ 1.87. IMPOSSIBLE for nonzero integer s except s=0,±1.
    # Type-22 has |sum| in {0,4}. |s|=4, cosine=4/sqrt(14)≈1.07 > 1? Wait ||v||=sqrt(8), ||u||=1.
    # I used unit u. Ganzhinov vectors as UNIT: v/sqrt(8). cosine = (a s)/sqrt(8) = s * sqrt(8/14)/sqrt(8) = s/sqrt(14).
    # |s|=4 => 4/sqrt(14)≈1.069 > 1 impossible? Inner product of unit vectors cannot exceed 1.
    # Type-22 vector (2,2,0^12)/sqrt(8) = (1,1,0^12)/sqrt(2). Sum of coords of integer = 4, 4/sqrt(14)≈1.069>1
    # means the all-equal UNIT vector vs type-22: 
    # <(2,2,0..)/√8 , (a,...,a)> = 4a/√8 = 4 sqrt(8/14)/sqrt(8) = 4/sqrt(14)≈1.069 > 1 — geometrically they are not both unit in a way... 
    # (a,...,a) with a=sqrt(8/14), ||u||^2 = 14*(8/14)=8, ||u||=sqrt(8). BOTH have norm sqrt(8).
    # <(2,2,0),(a..a)> = 4a = 4 sqrt(8/14)≈3.023, cosine=3.023/8=0.378 < 0.5. GOOD.
    # I mixed unit vs unnormalized. Correct: cosine = 4a / 8 = a/2 = sqrt(8/14)/2 = sqrt(2/14)=sqrt(1/7)≈0.378.
    # For w8, s = sum of 8 signs in {-8,-6,...,8}, ip = a s, cosine = a s / 8 = s sqrt(8/14)/8 = s / (sqrt(14)*sqrt(8)*sqrt(8)/sqrt(8) )
    # = s * sqrt(8/14) / 8 = s / (8 * sqrt(14/8)) = s / (8 * sqrt(7/4)) = s / (8 * sqrt(7)/2) = s / (4 sqrt(7))
    # Need |s| / (4 sqrt(7)) <= 1/2 iff |s| <= 2 sqrt(7) ≈ 5.291 iff |s| <= 5, so |s|<=4 (s even).
    s = g1932.astype(np.int16).sum(axis=1)
    bad = np.abs(s) >= 6
    n_bad = int(bad.sum())
    n_keep = int((~bad).sum())
    log(f"  1932 vectors with |coord-sum|>=6 (conflict with all-ones): {n_bad}; keep {n_keep}")
    # If we drop bad and add ±all-ones, net = n_keep + 2. Need > 1932 => n_keep > 1930 => n_bad < 2. Unlikely.
    log(f"  drop-conflicts+2 all-ones would give {n_keep + 2} (need >= 1933)")
    # Try other c in F_2^14: u_i = (-1)^{c_i} * a. Conflict when |signed sum| >= 6.
    rng = np.random.default_rng(3)
    best_keep = n_keep
    for k in range(400):
        if k == 0:
            signs = np.ones(14, dtype=np.int16)
        else:
            signs = rng.choice(np.array([-1, 1], dtype=np.int16), size=14)
        ss = (g1932.astype(np.int16) * signs).sum(axis=1)
        keep = int((np.abs(ss) <= 4).sum())
        if keep > best_keep:
            best_keep = keep
    log(f"  best keep among 400 random sign patterns: {best_keep}; with ±u => {best_keep+2}")
    if best_keep + 2 > 1932:
        log("  FAMILY E potential beat — constructing mixed-norm config")
    else:
        log("  FAMILY E ruled out: cannot keep enough of 1932 to net-gain from ±all-equal")


def family_f_bw16() -> None:
    log("FAMILY F: Barnes-Wall Λ16 coordinate 14D section")
    # AG(4,2) affine 3-flats as octads on 16 points.
    # Points of AG(4,2) = F_2^4.
    pts_ag = list(product((0, 1), repeat=4))
    assert len(pts_ag) == 16
    # Linear 3-dim subspaces of F_2^4: kernels of nonzero linear forms, up to scalar (only ±1=1).
    # Hyperplanes: {x : <a,x>=0} and the parallel {x : <a,x>=1} for a != 0, a up to scale.
    # Nonzero a, identify a ~ -a = a. 15 nonzero, 15 unique a, each gives 2 cosets, but each 3-flat
    # arises twice? 15 linear hyperplanes (through 0) and 15 affine. 30 flats.
    flats = []
    seen = set()
    for a in product((0, 1), repeat=4):
        if a == (0, 0, 0, 0):
            continue
        for b in (0, 1):
            octad = frozenset(i for i, x in enumerate(pts_ag) if (sum(ai * xi for ai, xi in zip(a, x)) % 2) == b)
            if octad not in seen and len(octad) == 8:
                seen.add(octad)
                flats.append(sorted(octad))
    log(f"  AG(4,2) 3-flats: {len(flats)}")
    # Delete coordinates 14,15. Keep octads contained in {0..13}.
    kept = [S for S in flats if max(S) <= 13]
    log(f"  octads inside first 14 coords: {len(kept)}")
    t22 = d14_type22()
    w8 = []
    for S in kept:
        for signs in product((-1, 1), repeat=8):
            if signs.count(-1) % 2 != 0:
                continue
            v = np.zeros(14, dtype=np.int8)
            for s, j in zip(signs, S):
                v[j] = s
            w8.append(v)
    if w8:
        w8a = np.stack(w8, axis=0)
        pts = np.unique(np.concatenate([t22, w8a], axis=0), axis=0)
    else:
        pts = t22
    res = verify_integer_equal_norm(pts)
    log(f"  BW16 even-sign section n={pts.shape[0]} ok={res.get('ok')} max={res.get('max_offdiag_inner')}")
    # Odd signs
    w8o = []
    for S in kept:
        for signs in product((-1, 1), repeat=8):
            if signs.count(-1) % 2 != 1:
                continue
            v = np.zeros(14, dtype=np.int8)
            for s, j in zip(signs, S):
                v[j] = s
            w8o.append(v)
    if w8o:
        pts_o = np.unique(np.concatenate([t22, np.stack(w8o)], axis=0), axis=0)
        res_o = verify_integer_equal_norm(pts_o)
        log(f"  BW16 odd-sign section n={pts_o.shape[0]} ok={res_o.get('ok')} max={res_o.get('max_offdiag_inner')}")


def family_g_witt() -> None:
    log("FAMILY G: shortened Witt S(5,8,24) even-sign (Leech-style) in 14D")
    # Generate Golay octads via the extended Golay code if possible; otherwise
    # use the MOG construction for a subset.
    # MiniMOG / hexacode construction of Golay:
    hexacode_words = []
    # Hexacode over F4={0,1,w,wbar} is a (6,3^2,4) code? 3^4=81 words? Standard hexacode has 64 words over F4, [6,3,4]_4.
    # We'll implement the binary Golay from quadratic residues of 23 plus a parity, then extract weight-8.
    # QR-23: quadratic residues mod 23.
    qr = {pow(i, 2, 23) for i in range(1, 23)}
    # cyclic QR code of length 23, extend.
    # NQRcode: basis = cyclic shifts of QR indicator plus all-ones? The quadratic residue code.
    # Simpler: generate the cyclic code from the QR polynomial.
    n = 23
    qr_ind = np.zeros(n, dtype=np.int8)
    for r in qr:
        qr_ind[r] = 1
    # Generator: shifts of qr_ind, plus all-ones. This generates the Golay [23,12] if qr includes 0 or not.
    # Standard: the binary QR code of length 23 has generator the polynomial with 1's at QR and infinity.
    rows = [qr_ind]
    for s in range(1, 23):
        rows.append(np.roll(qr_ind, s))
    rows.append(np.ones(23, dtype=np.int8))
    M = np.stack(rows, axis=0) % 2
    # row-reduce to get a spanning set, then enumerate 2^k is too big if k=12 (4096 ok actually).
    # Gaussian eliminate over F2
    A = M.copy()
    m, n23 = A.shape
    rank_rows = []
    used_col = set()
    r = 0
    for c in range(n23):
        pivot = None
        for i in range(r, m):
            if A[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(m):
            if i != r and A[i, c] == 1:
                A[i] = (A[i] + A[r]) % 2
        rank_rows.append(r)
        used_col.add(c)
        r += 1
        if r == m:
            break
    basis = A[:r]
    k = basis.shape[0]
    log(f"  QR-23 span rank={k}")
    if k > 14:
        # enumerating 2^k may be large; for k=12, 4096 is fine
        pass
    code23 = []
    for bits in range(1 << k):
        v = np.zeros(23, dtype=np.int8)
        for i in range(k):
            if (bits >> i) & 1:
                v ^= basis[i]
        code23.append(v)
    code23 = np.stack(code23, axis=0)
    # Extend to 24 by overall parity.
    par = code23.sum(axis=1) % 2
    code24 = np.concatenate([code23, par.reshape(-1, 1)], axis=1)
    wts = code24.sum(axis=1)
    octads = code24[wts == 8]
    log(f"  Golay length-24 words={code24.shape[0]} octads={octads.shape[0]} (expect 759)")
    # Shorten: keep octads with zeros in last 10 coordinates, drop those coords → 14D.
    # Try several deleted 10-sets: last 10, and a few random.
    t22 = d14_type22()
    best = 0
    rng = np.random.default_rng(4)
    delete_sets = [tuple(range(14, 24))]
    # also delete 10 from 0..23
    for _ in range(8):
        delete_sets.append(tuple(sorted(rng.choice(24, size=10, replace=False).tolist())))
    for dset in delete_sets:
        keep = [i for i in range(24) if i not in dset]
        assert len(keep) == 14
        good = []
        for row in octads:
            if all(row[j] == 0 for j in dset):
                good.append(row[list(keep)])
        if not good:
            n = 364
            log(f"  delete {dset[:3]}... octads-in-section=0 n=364")
            continue
        C = np.stack(good, axis=0)
        # even signs on each support
        w8 = []
        for supp_bits in C:
            S = [j for j in range(14) if supp_bits[j]]
            if len(S) != 8:
                continue
            for signs in product((-1, 1), repeat=8):
                if signs.count(-1) % 2:
                    continue
                v = np.zeros(14, dtype=np.int8)
                for s, j in zip(signs, S):
                    v[j] = s
                w8.append(v)
        pts = np.unique(np.concatenate([t22, np.stack(w8)], axis=0), axis=0) if w8 else t22
        res = verify_integer_equal_norm(pts)
        log(f"  section octads={C.shape[0]} n={pts.shape[0]} ok={res.get('ok')} max={res.get('max_offdiag_inner')}")
        best = max(best, int(pts.shape[0]) if res.get("ok") else 0)
        if res.get("ok") and pts.shape[0] > 1932:
            save_best(pts, "witt_section", "FAMILY G")
            return
    log(f"  FAMILY G best verified={best}")


def family_h_holes(g1932: np.ndarray, n_starts: int = 40) -> None:
    log("FAMILY H: numerical hole search on 1932 (float probe only; success requires algebraization)")
    G = g1932.astype(np.float64) / np.sqrt(8.0)
    rng = np.random.default_rng(5)
    best = 1.0
    best_x = None
    for s in range(n_starts):
        x = rng.normal(size=14)
        x /= np.linalg.norm(x)
        # minimize max_i <x, g_i> by a simple subgradient / softmax
        for it in range(400):
            dots = G @ x
            # softmax weights on largest inner products
            t = 80.0
            m = dots.max()
            w = np.exp(t * (dots - m))
            w /= w.sum()
            grad = G.T @ w
            # project gradient to tangent
            grad = grad - x * (x @ grad)
            x = x - 0.08 * grad
            x /= np.linalg.norm(x)
        mx = float((G @ x).max())
        if mx < best:
            best = mx
            best_x = x.copy()
        if (s + 1) % 10 == 0:
            log(f"  starts {s+1}: best max-inner={best:.6f} (need <= 0.5)")
    log(f"  FAMILY H best numerical max-inner={best:.6f}")
    if best <= 0.5 + 1e-9 and best_x is not None:
        log("  numerical hole found — attempting to snap to integer/rational shell")
        # Try to identify as a weight-8 or type-214 direction
        # Scale so largest coords look like 2 or 1
        for scale in (np.sqrt(8), 2.0, np.sqrt(2), np.sqrt(12), np.sqrt(24)):
            z = best_x * scale
            snapped = np.rint(z)
            if np.linalg.norm(z - snapped) < 0.15:
                v = snapped.astype(np.int8)
                if int(v @ v) == 0:
                    continue
                trial = np.unique(np.concatenate([g1932, v.reshape(1, -1), (-v).reshape(1, -1)]), axis=0)
                res = verify_integer_equal_norm(trial)
                log(f"  snap scale={scale} n={trial.shape[0]} ok={res.get('ok')} {res}")
                if res.get("ok") and trial.shape[0] > 1932:
                    save_best(trial, "numerical_hole_snapped", "FAMILY H")
                    return
        log("  hole did not snap to the integer shells tried")
    else:
        log("  no numerical hole with max-inner <= 1/2 (1932 appears locally maximal on the sphere among random probes)")


def analyze_supports(g1932: np.ndarray) -> None:
    w8, t22 = split_types(g1932)
    supps = []
    for v in w8:
        supp = tuple(i for i in range(14) if v[i] != 0)
        supps.append(supp)
    c = Counter(supps)
    log(f"ANALYZE: {len(c)} distinct 8-supports among {w8.shape[0]} weight-8 vectors")
    sizes = Counter(c.values())
    log(f"  vectors-per-support histogram: {dict(sorted(sizes.items()))}")
    keys = list(c.keys())
    inter = Counter()
    for a, b in combinations(keys, 2):
        inter[len(set(a) & set(b))] += 1
    log(f"  pairwise support intersections: {dict(sorted(inter.items()))}")


def main() -> None:
    CFG.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("")
    log("=== dim14 search start ===")
    log("LIVE RECORD: 1932 (Ganzhinov 2025) / upper 3174  https://cohn.mit.edu/kissing-numbers")
    g = ganzhinov_1932()
    res = verify_integer_equal_norm(g)
    log(f"baseline Ganzhinov verified: {json.dumps(res)}")
    np.save(CFG / "ganzhinov_1932.npy", g)
    np.savetxt(CFG / "ganzhinov_1932.txt", g, fmt="%d", delimiter=",")
    save_best(g, "ganzhinov_baseline", "reproduced arXiv:2207.08266 §5.5; not progress")
    analyze_supports(g)
    family_a_extra_w8(g)
    family_e_allones(g)
    family_f_bw16()
    family_g_witt()
    family_d_type214(g)
    family_h_holes(g)
    family_b_swaps(g, max_candidates=40000)
    family_c_greedy(n_trials=8)
    log("=== dim14 search pass finished ===")


if __name__ == "__main__":
    main()
