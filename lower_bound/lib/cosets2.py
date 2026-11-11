r"""Fast coset solving.

For a constrained pair (S,T) let A = {a in V^perp : a subset of S&T}\{0}.
Requirement: <a, c_S + c_T> = 1 for SOME a in A.
Writing c_S by its syndrome s_S in F_2^(n-d), each a becomes a functional
lam in F_2^(n-d) and the requirement is: par(lam & (s_S ^ s_T)) = 1 for some
lam in Lam.  |Lam| = 1  ->  a linear equation (checked exactly by elimination);
|Lam| > 1  ->  a disjunction, handled by search inside the solution space.
"""
def pc(x): return bin(x).count('1')
def par(x): return pc(x) & 1

def perp_basis(V, n):
    B = []
    for a in range(1 << n):
        if all(par(a & v) == 0 for v in V):
            x = a
            for b in B: x = min(x, x ^ b)
            if x: B.append(x); B.sort(reverse=True)
    return B

def constraints(sups, V, n):
    """returns (pairs, k) with pairs = [(i, j, [lam...])] or None if impossible"""
    B = perp_basis(V, n); k = len(B)
    full = (1 << n) - 1
    Pall = []
    for a in range(1 << n):
        if all(par(a & v) == 0 for v in V) and a: Pall.append(a)
    coord = {}
    for a in Pall:
        x = a; lam = 0
        for idx, b in enumerate(B):
            if x ^ b < x: x ^= b; lam |= 1 << idx
        if x == 0: coord[a] = lam
    out = []
    for i in range(len(sups)):
        for j in range(i + 1, len(sups)):
            X = sups[i] & sups[j]; t = pc(X)
            if t <= 4: continue
            if t >= 7: return None, k
            A = [a for a in Pall if not (a & (full ^ X))]
            if not A: return None, k
            out.append((i, j, [coord[a] for a in A]))
    return out, k

def solve(sups, V, n, seed=0, restarts=60, sweeps=4000):
    import random
    cons, k = constraints(sups, V, n)
    if cons is None: return None
    m = len(sups)
    # ---- exact feasibility of the forced (single-lambda) equations ----
    piv = {}
    def addrow(row, rhs):
        while row:
            h = row.bit_length() - 1
            if h in piv: pr, prhs = piv[h]; row ^= pr; rhs ^= prhs
            else: piv[h] = (row, rhs); return True
        return rhs == 0
    for (i, j, L) in cons:
        if len(L) == 1:
            lam = L[0]; row = 0
            for b in range(k):
                if lam >> b & 1: row ^= (1 << (i * k + b)) ^ (1 << (j * k + b))
            if not addrow(row, 1): return None          # provably infeasible
    # ---- local search for a full solution ----
    rng = random.Random(seed)
    inc = [[] for _ in range(m)]
    for idx, (i, j, L) in enumerate(cons): inc[i].append(idx); inc[j].append(idx)
    KK = 1 << k
    for _ in range(restarts):
        s = [rng.randrange(KK) for _ in range(m)]
        def viol(idx):
            i, j, L = cons[idx]; d = s[i] ^ s[j]
            return not any(par(l & d) for l in L)
        bad = {q for q in range(len(cons)) if viol(q)}
        for _ in range(sweeps):
            if not bad: break
            q = rng.choice(tuple(bad)); i, j, _ = cons[q]
            node = i if rng.random() < 0.5 else j
            bestv, bestc = s[node], 1 << 30
            for _try in range(48):
                cand = rng.randrange(KK)
                old = s[node]; s[node] = cand
                cnt = sum(1 for w in inc[node] if viol(w))
                s[node] = old
                if cnt < bestc: bestc, bestv = cnt, cand
                if cnt == 0: break
            s[node] = bestv
            for w in inc[node]:
                if viol(w): bad.add(w)
                else: bad.discard(w)
        if not bad:
            B = perp_basis(V, n)
            rep = {}
            for c in range(1 << n):
                sy = 0
                for idx, b in enumerate(B):
                    if par(b & c): sy |= 1 << idx
                if sy not in rep: rep[sy] = c
            return [rep[x] for x in s]
    return None
