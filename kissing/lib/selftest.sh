#!/bin/bash
# Repo self-checks that should finish in about a minute from a clean tree.
# Run from the repo root.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

# NOTE: mis8 rewrites kissing/logs/mis8_n14_s1.txt as a side effect, so this
# self-check leaves that tracked file modified.  Restore it with
# `git checkout -- kissing/logs/mis8_n14_s1.txt` before committing.
echo "== mis8: load Ganzhinov 1568 with zero violations =="
python3 - <<'PY'
import numpy as np, sys
from pathlib import Path
V = np.loadtxt("kissing/dim14/configs/ganzhinov_1932.txt", delimiter=",", dtype=np.int64)
w8 = V[np.count_nonzero(V, axis=1) == 8]
assert len(w8) == 1568, len(w8)
n = 14
sup, idxof = [], [-1] * (1 << n)
for m in range(1 << n):
    if bin(m).count("1") == 8:
        idxof[m] = len(sup)
        sup.append(m)
lines = []
for v in w8:
    m = 0
    bits = []
    for b in range(n):
        if v[b] != 0:
            m |= 1 << b
            bits.append(b)
    cs = 0
    for k, b in enumerate(bits):
        if v[b] < 0:
            cs |= 1 << k
    lines.append(f"{idxof[m]} {cs}")
Path("/tmp/g14_mis8_seed.txt").write_text("\n".join(lines) + "\n")
print(f"wrote {len(lines)} vertices")
PY
# 0 seconds: load the seed and report its size (conflicts would drop nsol).
out=$(./kissing/lib/mis8 14 0 1 /tmp/g14_mis8_seed.txt 2>/tmp/mis8_err.txt)
echo "$out"
echo "$(cat /tmp/mis8_err.txt)"
echo "$out" | grep -q 'best_weight8=1568'
grep -q 'seeded solution size 1568' /tmp/mis8_err.txt
echo "mis8 OK"

echo "== gcode: rebuild a valid 1932 from V + supports + cosets =="
python3 - <<'PY'
import json, random, sys
import numpy as np
sys.path.insert(0, "kissing/lib")
from gcode import Ctx, Elim, pair_rows, solve_signs, build, check

n = 14
PCT = np.array([bin(x).count("1") for x in range(1 << n)], dtype=np.int8)
SUPS = np.array([m for m in range(1 << n) if bin(m).count("1") == 8], dtype=np.int32)
d = json.load(open("kissing/lib/g14_struct.json"))
sups = [sum(1 << i for i in s) for s in d["supports"]]
assert len(sups) == 49

def span(g):
    B = []
    for x in g:
        for b in B:
            x = min(x, x ^ b)
        if x:
            B.append(x)
            B.sort(reverse=True)
    return B

def inspan(B, v):
    for b in B:
        if v ^ b < v:
            v ^= b
    return v == 0

def bits(m):
    return [b for b in range(n) if m >> b & 1]

lift = [[sum(1 << bits(S)[k] for k in range(8) if w >> k & 1) for w in C]
        for S, C in zip(sups, d["codes"])]
U = [span([w ^ L[0] for w in L]) for L in lift]
V = [x for x in range(1 << n) if all(inspan(U[i], x & sups[i]) for i in range(49))]
ctx = Ctx(V, n, PCT, SUPS)
el = Elim(); F = []
for S in sups:
    rows = pair_rows(ctx, F, S)
    assert rows is not None
    for r in rows:
        assert el.add(r, 1)
    F.append(S)
rng = random.Random(0)
c = solve_signs(ctx, F, el, rng, tries=400)
assert c is not None, "coset solve failed"
W = build(F, V, c, n)
ok, mx = check(W, n)
print(f"weight8={len(W)} ok={ok} max_inner={mx} Vdim={ctx.d} supports={len(F)}")
assert ok and len(W) == 1568 and mx <= 4
print(f"total_with_Droots={len(W) + 4 * n * (n - 1) // 2}")
print("gcode OK")
PY

echo "== clique: dim-14 design 49, dim-13 21 =="
# allowed intersections {4,6} => bits 4 and 6 set => 0x50 = 80
c14=$(./kissing/lib/clique 14 8 80 12 1 2>/tmp/clique14_err.txt)
echo "$c14" | head -1
c13=$(./kissing/lib/clique 13 8 80 8 1 2>/tmp/clique13_err.txt)
echo "$c13" | head -1
echo "$c14" | grep -q 'max_clique_found=49'
echo "$c13" | grep -q 'max_clique_found=21'
echo "clique OK"

echo "== riesz selftest (BLAS engrad vs reference loops) =="
# riesz/riesz2 are build products, not committed binaries, so build them here.
# The BLAS-free checks above have already run by this point, so a machine
# without OpenBLAS still gets everything it can.
if [ ! -x kissing/lib/riesz2 ]; then
  if ! make -C kissing/lib riesz2; then
    echo "SKIP riesz selftest: could not build riesz2 (see message above)" >&2
    echo "all BLAS-free self-checks passed"
    exit 0
  fi
fi
KISS_SELFTEST=1 ./kissing/lib/riesz2

echo "all self-checks passed"
