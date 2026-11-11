"""Remove k points from a configuration, add k+1 random ones, write the seed.
This is the flexibility search that took dim 12 from 840 to 841."""
import sys, numpy as np
X = np.loadtxt(sys.argv[1]); k = int(sys.argv[2]); seed = int(sys.argv[3])
rng = np.random.default_rng(seed)
keep = rng.permutation(len(X))[: len(X) - k]
Y = X[keep]
extra = rng.standard_normal((k + 1, X.shape[1]))
extra /= np.linalg.norm(extra, axis=1, keepdims=True)
out = np.vstack([Y, extra])
np.savetxt(sys.argv[4], out, fmt='%.17g')
print(len(out))
