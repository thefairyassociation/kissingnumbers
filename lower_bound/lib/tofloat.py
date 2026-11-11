"""Dump an exact config JSON as unit-normalised float rows (for the C optimiser)."""
import sys, json, math, re
sys.path.insert(0, 'lower_bound/lib')
from verify_exact import load
d, V, D = load(sys.argv[1])
out = []
for v in V:
    x = [float(a) + float(b) * math.sqrt(D) for a, b in v]
    s = math.sqrt(sum(c * c for c in x))
    out.append([c / s for c in x])
extra = int(sys.argv[3]) if len(sys.argv) > 3 else 0
import random
random.seed(12345)
for _ in range(extra):
    x = [random.gauss(0, 1) for _ in range(d["dimension"])]
    s = math.sqrt(sum(c * c for c in x)); out.append([c / s for c in x])
with open(sys.argv[2], 'w') as f:
    for r in out: f.write(' '.join('%.17g' % c for c in r) + '\n')
print(f"wrote {len(out)} rows of dim {d['dimension']} to {sys.argv[2]}")
