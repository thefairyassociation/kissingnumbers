"""Global-linear-code framework for weight-8 kissing configurations in Z^n.

A configuration is described by
  * a linear code V <= F_2^n,
  * a family S of 8-subsets ("supports"), each 'good': every nonzero codeword
    meets S in >= 2 points  (<=> V|S is injective with min weight >= 2),
  * a coset representative c_S per support; the signs used on S are
    (c_S + V)|_S,  so |C_S| = |V|.

Pair (S,T) with t = |S^T|:
  t <= 4 : always fine
  t in {5,6} : need V|_(S^T) proper, i.e. some nonzero v in V with v&(S^T)=0,
               and then (c_S+c_T)|_(S^T) not in V|_(S^T)
  t = 7,8 : need distance >= 2 between the projections (handled separately)
Total weight-8 vectors = |S| * |V|;  add 4*C(n,2) D-roots for the kissing count.
"""
import itertools, random, sys
from functools import lru_cache

def popcount(x): return bin(x).count('1')

def span(gens):
    B = []
    for g in gens:
        for b in B: g = min(g, g ^ b)
        if g: B.append(g); B.sort(reverse=True)
    return B

def words(basis):
    out = [0]
    for b in basis: out += [x ^ b for x in out]
    return out

def good_supports(V, n, k=8):
    """8-subsets meeting every nonzero codeword in >= 2 points."""
    nz = [v for v in V if v]
    out = []
    for S in itertools.combinations(range(n), k):
        m = sum(1 << i for i in S)
        if all(popcount(v & m) >= 2 for v in nz): out.append(m)
    return out

def kernel_table(V, n):
    """hasker[X] = some nonzero codeword avoids the coordinate set X."""
    has = bytearray(1 << n)
    full = (1 << n) - 1
    for v in V:
        if not v: continue
        c = full ^ v
        sub = c
        while True:
            has[sub] = 1
            if sub == 0: break
            sub = (sub - 1) & c
    return has

def compat(masks, V, n, allow_t):
    """adjacency bitsets over the good-support list"""
    has = kernel_table(V, n)
    m = len(masks)
    adj = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            X = masks[i] & masks[j]; t = popcount(X)
            ok = (t <= 4) or (t in (5, 6) and t in allow_t and has[X])
            if ok: adj[i] |= 1 << j; adj[j] |= 1 << i
    return adj

def maxclique(adj, tries=600, rng=None):
    rng = rng or random.Random(0)
    m = len(adj); best = []
    full = (1 << m) - 1
    for _ in range(tries):
        cand = full; cur = []
        while cand:
            bits = [i for i in range(m) if cand >> i & 1]
            v = rng.choice(bits) if rng.random() < 0.5 else max(bits, key=lambda x: popcount(adj[x] & cand))
            cur.append(v); cand &= adj[v]
        if len(cur) > len(best): best = cur
    return best

def assign_cosets(sel, V, n, rng, tries=400):
    """choose c_S in F_2^n so that (c_S+c_T)|_(S^T) not in V|_(S^T) for the
    constrained pairs; returns list of reps or None."""
    Vs = set(V)
    pairs = []
    for a in range(len(sel)):
        for b in range(a + 1, len(sel)):
            X = sel[a] & sel[b]
            if popcount(X) in (5, 6): pairs.append((a, b, X))
    def ok(c, X, Vset):
        return all((c ^ v) & X for v in Vset) if False else ((c & X) not in {v & X for v in Vset})
    proj = {}
    for _, _, X in pairs:
        if X not in proj: proj[X] = {v & X for v in V}
    for _ in range(tries):
        reps = [0] * len(sel); order = list(range(1, len(sel)))
        rng.shuffle(order); good = True
        for idx in order:
            choices = list(range(1 << n)); rng.shuffle(choices)
            placed = False
            for c in choices[:4000]:
                if all(((c ^ reps[a]) & X) not in proj[X]
                       for a, b, X in pairs if b == idx and (a == 0 or a in order[:order.index(idx)])):
                    pass
                good2 = True
                for a, b, X in pairs:
                    if b == idx and (a == 0 or order.index(a) < order.index(idx)):
                        if ((c ^ reps[a]) & X) in proj[X]: good2 = False; break
                    if a == idx and (b == 0 or (b in order and order.index(b) < order.index(idx))):
                        if ((c ^ reps[b]) & X) in proj[X]: good2 = False; break
                if good2: reps[idx] = c; placed = True; break
            if not placed: good = False; break
        if good: return reps
    return None
