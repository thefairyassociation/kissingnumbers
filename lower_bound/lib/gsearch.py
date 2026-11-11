"""Search: global linear code V <= F_2^n + family of 8-supports.

Configuration = 4*C(n,2) D-roots + |S| * 2^dim(V) weight-8 vectors.
Conditions (S,T supports, X = S&T, t = |X|):
  goodness : every nonzero v in V has |v & S| >= 2   (for every support S)
  t <= 4   : free
  t in {5,6} : dim(V|X) < t,  i.e. dim(V) - kerdim(X) < t
  t >= 7   : rejected here (needs a stronger coset condition)
plus per-support coset reps c_S with (c_S+c_T)|X not in V|X for constrained pairs.
"""
import itertools, random, sys

def pc(x): return bin(x).count('1')

def code_words(basis):
    out = [0]
    for b in basis: out += [x ^ b for x in out]
    return out

def kerdims(V, n):
    """kerdim[X] = log2 |{v in V : v & X == 0}| for every coordinate set X"""
    K = [0] * (1 << n)
    for X in range(1 << n):
        c = 0
        for v in V:
            if not (v & X): c += 1
        K[X] = c.bit_length() - 1
    return K

def kerdims_fast(V, n):
    """same, via zeta transform over subsets: cnt[X] = #{v in V : v subset of X}"""
    cnt = [0] * (1 << n)
    for v in V: cnt[v] += 1
    for b in range(n):
        for X in range(1 << n):
            if X >> b & 1: cnt[X] += cnt[X ^ (1 << b)]
    full = (1 << n) - 1
    return [ (cnt[full ^ X]).bit_length() - 1 for X in range(1 << n) ]

def good_supports(V, n, k=8):
    nz = [v for v in V if v]
    return [m for m in range(1 << n) if pc(m) == k and all(pc(v & m) >= 2 for v in nz)]

def build_adj(G, V, n, d, K, allow7=False):
    m = len(G); adj = [0] * m
    for i in range(m):
        Gi = G[i]
        for j in range(i + 1, m):
            X = Gi & G[j]; t = pc(X)
            if t <= 4: ok = True
            elif t in (5, 6): ok = (d - K[X]) < t
            else: ok = False
            if ok: adj[i] |= 1 << j; adj[j] |= 1 << i
    return adj

def greedy_clique(adj, rng, tries):
    m = len(adj); best = []
    for _ in range(tries):
        cand = (1 << m) - 1; cur = []
        while cand:
            bl = cand.bit_length(); pool = []
            x = cand
            while x:
                b = x & -x; pool.append(b.bit_length() - 1); x ^= b
            v = rng.choice(pool) if rng.random() < 0.6 else max(pool, key=lambda z: pc(adj[z] & cand))
            cur.append(v); cand &= adj[v]
        if len(cur) > len(best): best = cur
    return best

def score_code(basis, n, rng, tries=40):
    V = code_words(basis); d = len(basis)
    if len(set(V)) != 1 << d: return 0, None, None, None
    G = good_supports(V, n)
    if len(G) < 4: return 0, None, None, None
    K = kerdims_fast(V, n)
    adj = build_adj(G, V, n, d, K)
    cl = greedy_clique(adj, rng, tries)
    return len(cl) * (1 << d), [G[i] for i in cl], V, d
