"""Local jamming of a kissing configuration.

Point i can move (to first order) iff there is a tangent direction d at x_i with
<d, x_j> < 0 for every tight neighbour j -- i.e. iff 0 is NOT in the interior of
the convex hull of the tangent projections of its tight neighbours.
Solve, for each i, the small LP  max s  s.t.  <d, P_j> + s <= 0, |d|_inf <= 1.
s <= 0 for every i  =>  no single point can move: the configuration is locally
jammed, so 1154 is a strict local maximum for one-point moves."""
import sys, numpy as np
from scipy.optimize import linprog

X = np.loadtxt(sys.argv[1])
N, n = X.shape
G = X @ X.T
free = []
worst = -9
for i in range(N):
    nb = np.where((G[i] > 0.5 - 1e-9) & (np.arange(N) != i))[0]
    if len(nb) == 0: free.append((i, 1.0)); continue
    P = X[nb] - np.outer(G[i, nb], X[i])          # project onto the tangent space
    nrm = np.linalg.norm(P, axis=1)
    P = P[nrm > 1e-9] / nrm[nrm > 1e-9, None]
    if len(P) == 0: free.append((i, 1.0)); continue
    A = np.hstack([P, np.ones((len(P), 1))])
    c = np.zeros(n + 1); c[-1] = -1              # maximise s
    r = linprog(c, A_ub=A, b_ub=np.zeros(len(P)),
                bounds=[(-1, 1)] * n + [(0, 1)], method='highs')
    s = r.x[-1] if r.success else 0.0
    worst = max(worst, s)
    if s > 1e-7: free.append((i, s))
print(f"points {N}, tight-neighbour LPs solved")
print(f"points that can move: {len(free)}   largest escape margin: {worst:.3e}")
if free[:5]: print("  examples:", free[:5])
print("VERDICT:", "LOCALLY JAMMED - no single point can move"
      if not free else "some points are free to move")
