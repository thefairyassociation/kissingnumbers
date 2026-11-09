"""Remove k points from a configuration and re-insert k+1 at the DEEPEST HOLES
of what remains (LP), rather than at random.  Much better starting points for
the continuation optimiser than random re-insertion.

usage: shake2.py <base floats> <k> <seed> <out>"""
import sys, numpy as np
sys.path.insert(0, 'lower_bound/lib')
from holes import max_norm_over_P

X = np.loadtxt(sys.argv[1]); k = int(sys.argv[2]); seed = int(sys.argv[3])
rng = np.random.default_rng(seed)
keep = rng.permutation(len(X))[: len(X) - k]
Y = list(X[keep])
for _ in range(k + 1):
    nb, u = max_norm_over_P(np.array(Y), starts=14, seed=int(rng.integers(1 << 30)))
    if u is None or np.linalg.norm(u) < 1e-9:
        u = rng.standard_normal(X.shape[1])
    Y.append(u / np.linalg.norm(u))
np.savetxt(sys.argv[4], np.array(Y), fmt='%.17g')
print(len(Y))
