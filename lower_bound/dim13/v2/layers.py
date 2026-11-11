"""Decompose ZE99's 1154 into its layers and test whether the height-+-1/2
   layer (a 1/3-code in R^12) can be enlarged.

   N(13) = 2 + |A| + |B_+| + |B_-| ; A a 1/2-code at height 0, B_+- 1/3-codes at
   height +-1/2, cross condition cos(A,B) <= 1/sqrt(3), none between B_+ and B_-.
   ZE99 = 2 + 816 + 168 + 168 = 1154.  ONE extra point in B gives 1156."""
import os, tempfile
# Intermediate arrays are shared between layers.py, completeB.py and greedyB.py.
SCRATCH = os.environ.get("KISS_SCRATCH", os.path.join(tempfile.gettempdir(), "kissing-scratch"))
os.makedirs(SCRATCH, exist_ok=True)
import sys, json, math, numpy as np
sys.path.insert(0, 'lower_bound/lib')
from verify_exact import load

d, V, D = load('lower_bound/dim13/configs/ze99_1154_exact.json')
X = np.array([[float(a) + float(b) * math.sqrt(D) for a, b in v] for v in V])
X = X / np.linalg.norm(X, axis=1, keepdims=True)
h = X[:, 12]
poles = X[np.abs(h) > 0.9]
A = X[np.abs(h) < 1e-9][:, :12]
Bp = X[(h > 0.4) & (h < 0.9)]
Bm = X[(h < -0.4) & (h > -0.9)]
print(f"layers: poles {len(poles)}  A (h=0) {len(A)}  B+ {len(Bp)}  B- {len(Bm)}  "
      f"total {len(poles)+len(A)+len(Bp)+len(Bm)}")
print("distinct heights:", sorted({round(v, 6) for v in h.tolist()}))
Bpu = Bp[:, :12] / np.linalg.norm(Bp[:, :12], axis=1, keepdims=True)
Bmu = Bm[:, :12] / np.linalg.norm(Bm[:, :12], axis=1, keepdims=True)
def mx(M):
    G = M @ M.T; np.fill_diagonal(G, -9); return G.max()
print(f"max cos inside A  = {mx(A):.6f}  (needs <= 1/2 = 0.5)")
print(f"max cos inside B+ = {mx(Bpu):.6f}  (needs <= 1/3 = {1/3:.6f})")
print(f"max cos A x B+    = {(A @ Bpu.T).max():.6f}  (needs <= 1/sqrt3 = {1/math.sqrt(3):.6f})")
print(f"B+ and B- unit sets identical: {np.allclose(np.sort(Bpu,axis=0), np.sort(Bmu,axis=0))}")
np.save(os.path.join(SCRATCH, 'A12.npy'), A)
np.save(os.path.join(SCRATCH, 'B12.npy'), Bpu)
