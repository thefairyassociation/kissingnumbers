"""First-order rigidity of a kissing configuration.

Infinitesimal motions xdot must satisfy  <xdot_i, x_i> = 0 (stay on the sphere)
and, for every tight pair (<x_i,x_j> = 1/2),
      <xdot_i, x_j> + <x_i, xdot_j> <= 0.
Rotations (xdot_i = W x_i, W antisymmetric) satisfy these with equality.
Minimise the total derivative sum over tight pairs subject to |xdot| <= 1:
the optimum is 0 exactly when NO motion can open any tight contact, i.e. the
configuration is first-order rigid and its size is a strict local maximum."""
import sys, math, time, numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
sys.path.insert(0, 'lower_bound/lib')

X = np.loadtxt(sys.argv[1])
N, n = X.shape
G = X @ X.T
tight = np.argwhere((np.triu(G, 1) > 0.5 - 1e-9) & (np.triu(np.ones_like(G), 1) > 0))
print(f"config {N}x{n}, tight pairs {len(tight)}")

rows, cols, vals = [], [], []
for r, (i, j) in enumerate(tight):
    for k in range(n):
        rows += [r, r]; cols += [i * n + k, j * n + k]; vals += [X[j, k], X[i, k]]
A_ub = coo_matrix((vals, (rows, cols)), shape=(len(tight), N * n)).tocsr()
b_ub = np.zeros(len(tight))
er, ec, ev = [], [], []
for i in range(N):
    for k in range(n):
        er.append(i); ec.append(i * n + k); ev.append(X[i, k])
A_eq = coo_matrix((ev, (er, ec)), shape=(N, N * n)).tocsr()
b_eq = np.zeros(N)
c = np.asarray(A_ub.sum(axis=0)).ravel()          # minimise sum of contact derivatives
t0 = time.time()
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
              bounds=[(-1, 1)] * (N * n), method='highs')
print(f"LP status: {res.message}  objective = {res.fun:.6e}  [{time.time()-t0:.0f}s]")
if res.success:
    xd = res.x.reshape(N, n)
    d = A_ub @ res.x
    print(f"most-opened contact derivative = {d.min():.6e}  (0 => first-order rigid)")
    print(f"||xdot||_inf = {np.abs(xd).max():.4f}, ||xdot||_2 = {np.linalg.norm(xd):.4f}")
    print("VERDICT:", "FIRST-ORDER RIGID (no contact can open)" if d.min() > -1e-7
          else "FLEXIBLE: some contact opens -> deformation search is worthwhile")
