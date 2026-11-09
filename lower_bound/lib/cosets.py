"""Coset representatives for the global-code framework (all F_2 linear algebra).

Pair (S,T), X = S&T.  Sign codes (c_S+V)|_X and (c_T+V)|_X must be disjoint,
i.e.  <a, c_S + c_T> = 1  for at least one a in A_X = {a in V^perp : a subset X}.
|A_X| = 2^(t - dim V|X) - 1 nonzero choices; when there is exactly one, the
requirement is a linear equation over F_2.
"""
def pc(x): return bin(x).count('1')

def perp(V, n):
    """all vectors orthogonal to every codeword of V"""
    return [a for a in range(1 << n) if all(pc(a & v) % 2 == 0 for v in V)]

class F2System:
    """rows over F_2 with right-hand sides; incremental elimination"""
    def __init__(self, nv): self.nv = nv; self.piv = {}
    def add(self, row, rhs):
        while row:
            h = row.bit_length() - 1
            if h in self.piv:
                pr, prhs = self.piv[h]; row ^= pr; rhs ^= prhs
            else:
                self.piv[h] = (row, rhs); return True
        return rhs == 0                      # 0 = 0 fine, 0 = 1 contradiction
    def solve(self):
        y = [0] * self.nv
        for h in sorted(self.piv, reverse=True):
            row, rhs = self.piv[h]
            s = rhs; rr = row ^ (1 << h)
            while rr:
                b = rr & -rr; rr ^= b; s ^= y[b.bit_length() - 1]
            y[h] = s
        return y

def solve_cosets(sups, V, n, verbose=False):
    """returns list of coset reps c_S (or None if the linear part is infeasible)"""
    P = [a for a in perp(V, n) if a]
    m = len(sups)
    sysm = F2System(m * n)
    hard = []
    for i in range(m):
        for j in range(i + 1, m):
            X = sups[i] & sups[j]; t = pc(X)
            if t <= 4: continue
            A = [a for a in P if not (a & ~X & ((1 << n) - 1))]
            if not A: return None, "pair (%d,%d) has empty annihilator" % (i, j)
            if len(A) == 1:
                a = A[0]; row = 0
                for b in range(n):
                    if (a >> b) & 1: row ^= (1 << (i * n + b)) ^ (1 << (j * n + b))
                if not sysm.add(row, 1): return None, "linear system inconsistent"
            else:
                hard.append((i, j, A))
    y = sysm.solve()
    c = [sum(y[i * n + b] << b for b in range(n)) for i in range(m)]
    # verify every constrained pair, including the disjunctive ones
    bad = []
    for i in range(m):
        for j in range(i + 1, m):
            X = sups[i] & sups[j]; t = pc(X)
            if t <= 4: continue
            A = [a for a in P if not (a & ~X & ((1 << n) - 1))]
            if not any(pc(a & (c[i] ^ c[j])) % 2 for a in A): bad.append((i, j))
    return c, bad

def build_vectors(sups, V, c, n):
    """explicit integer weight-8 vectors: +1 where the sign bit is 0, -1 where 1"""
    out = []
    for S, cs in zip(sups, c):
        idx = [b for b in range(n) if S >> b & 1]
        seen = set()
        for v in V:
            w = (cs ^ v) & S
            if w in seen: continue
            seen.add(w)
            vec = [0] * n
            for b in idx: vec[b] = -1 if (w >> b) & 1 else 1
            out.append(vec)
    return out

def syndrome_setup(V, n):
    """basis of V^perp and the syndrome map c -> F_2^(n-d)"""
    P = perp(V, n)
    B = []
    for a in P:
        x = a
        for b in B: x = min(x, x ^ b)
        if x: B.append(x); B.sort(reverse=True)
    return B                                    # len = n - dim V

def pair_forbidden(sups, V, n, B):
    """for each constrained pair: subspace M (as a set) of forbidden syndrome
    differences in F_2^len(B); condition is  s_i ^ s_j  not in M."""
    P = [a for a in perp(V, n) if a]
    k = len(B)
    full = (1 << n) - 1
    coord = {}
    for a in P:                                  # express a in the basis B
        x = a; lam = 0
        for idx, b in enumerate(B):
            if x ^ b < x: x ^= b; lam |= 1 << idx
        assert x == 0
        coord[a] = lam
    out = []
    for i in range(len(sups)):
        for j in range(i + 1, len(sups)):
            X = sups[i] & sups[j]; t = pc(X)
            if t <= 4: continue
            A = [a for a in P if not (a & (full ^ X))]
            if not A: return None
            lams = [coord[a] for a in A]
            M = set()
            for s in range(1 << k):
                if all(pc(s & l) % 2 == 0 for l in lams): M.add(s)
            out.append((i, j, frozenset(M)))
    return out

def solve_cosets_ls(sups, V, n, seed=0, iters=400000):
    """min-conflicts search for syndromes s_S with s_i ^ s_j not in M_ij"""
    import random
    rng = random.Random(seed)
    B = syndrome_setup(V, n); k = len(B)
    pf = pair_forbidden(sups, V, n, B)
    if pf is None: return None
    m = len(sups)
    inc = [[] for _ in range(m)]
    for idx, (i, j, M) in enumerate(pf): inc[i].append(idx); inc[j].append(idx)
    s = [rng.randrange(1 << k) for _ in range(m)]
    def viol(idx):
        i, j, M = pf[idx]; return (s[i] ^ s[j]) in M
    bad = {idx for idx in range(len(pf)) if viol(idx)}
    for it in range(iters):
        if not bad: break
        idx = rng.choice(tuple(bad))
        i, j, _ = pf[idx]
        node = i if rng.random() < 0.5 else j
        bestv, bestc = s[node], 10 ** 9
        for cand in range(1 << k):
            old = s[node]; s[node] = cand
            cnt = sum(1 for q in inc[node] if viol(q))
            s[node] = old
            if cnt < bestc or (cnt == bestc and rng.random() < 0.2): bestc, bestv = cnt, cand
        s[node] = bestv
        for q in inc[node]:
            if viol(q): bad.add(q)
            else: bad.discard(q)
    if bad: return None
    # lift syndromes back to representatives c_S
    import itertools
    rep = {}
    for c in range(1 << n):
        sy = 0
        for idx, b in enumerate(B):
            if pc(b & c) % 2: sy |= 1 << idx
        if sy not in rep: rep[sy] = c
        if len(rep) == 1 << k: break
    return [rep[x] for x in s]
