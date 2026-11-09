"""Global-linear-code framework for weight-8 kissing configurations in Z^n.

configuration = 4*C(n,2) D-roots  +  sum_S |V|  weight-8 vectors
supports S: every nonzero codeword of V meets S in >= 2 coordinates
pairs: t=|S&T| <= 4 free; t in {5,6} need some a in V^perp\\0 with supp(a) <= S&T,
       and then <a, c_S + c_T> = 1 for one such a; t >= 7 rejected.
"""
import random
import numpy as np

def pc(x): return bin(x).count('1')
def par(x): return pc(x) & 1

def code_words(basis):
    out = [0]
    for b in basis: out += [x ^ b for x in out]
    return out

class Ctx:
    def __init__(self, V, n, PCT, SUPS):
        self.n = n; self.V = V; self.d = len(V).bit_length() - 1
        self.PCT = PCT
        nz = [v for v in V if v]
        ok = np.ones(len(SUPS), dtype=bool)
        for v in nz: ok &= PCT[np.bitwise_and(SUPS, v)] >= 2
        self.good = SUPS[ok].tolist()
        allv = np.arange(1 << n, dtype=np.int32)
        keep = np.ones(1 << n, dtype=bool)
        for v in nz: keep &= (PCT[np.bitwise_and(allv, v)] & 1) == 0
        self.perp = allv[keep].tolist()
        B = []
        for a in self.perp:
            x = a
            for b in B: x = min(x, x ^ b)
            if x: B.append(x); B.sort(reverse=True)
        self.B = B; self.k = len(B)
        coord = {}
        for a in self.perp:
            x = a; lam = 0
            for i, b in enumerate(B):
                if x ^ b < x: x ^= b; lam |= 1 << i
            coord[a] = lam
        self.coord = coord
        cnt = np.zeros(1 << n, dtype=np.int32); lamx = np.zeros(1 << n, dtype=np.int32)
        for a in self.perp:
            if a: cnt[a] += 1; lamx[a] ^= coord[a]
        idx = np.arange(1 << n)
        for b in range(n):
            hi = idx[(idx >> b & 1) == 1]
            cnt[hi] += cnt[hi ^ (1 << b)]; lamx[hi] ^= lamx[hi ^ (1 << b)]
        self.cnt = cnt; self.lamx = lamx
        # representative of each syndrome class, for lifting back
        rep = {}
        for c in range(1 << n):
            sy = 0
            for i, b in enumerate(B):
                if par(b & c): sy |= 1 << i
            if sy not in rep: rep[sy] = c
        self.rep = rep

    def lams(self, X):
        """all functionals for the pair-set X (as lambda coordinates)"""
        c = getattr(self, '_lamcache', None)
        if c is None: c = self._lamcache = {}
        v = c.get(X)
        if v is None:
            full = (1 << self.n) - 1
            v = c[X] = [self.coord[a] for a in self.perp if a and not (a & (full ^ X))]
        return v

class Elim:
    __slots__ = ('piv',)
    def __init__(self, piv=None): self.piv = dict(piv) if piv else {}
    def copy(self): return Elim(self.piv)
    def add(self, row, rhs):
        p = self.piv
        while row:
            h = row.bit_length() - 1
            if h in p:
                pr, prhs = p[h]; row ^= pr; rhs ^= prhs
            else:
                p[h] = (row, rhs); return True
        return rhs == 0
    def sample(self, nv, rng):
        """random solution of the affine system (free variables randomised)"""
        y = [0] * nv
        pivset = self.piv
        for h in range(nv):
            if h in pivset: continue
            y[h] = rng.getrandbits(1)
        for h in sorted(pivset):
            row, rhs = pivset[h]
            s = rhs; rr = row ^ (1 << h)
            while rr:
                b = rr & -rr; rr ^= b; s ^= y[b.bit_length() - 1]
            y[h] = s
        return y

def pair_rows(ctx, F, S, rng=None):
    """Rows forced by adding support S to family F; None if structurally bad.

    A pair whose annihilator is 1-dimensional gives one linear equation.  When it
    is larger the requirement is a disjunction ("some functional separates the two
    cosets"); passing an rng picks one functional at random and imposes it as an
    equation, which is a sufficient - and far more tractable - condition."""
    rows = []; k = ctx.k; j = len(F)
    for i, T in enumerate(F):
        X = S & T; t = ctx.PCT[X]
        if t <= 4: continue
        if t >= 7 or ctx.cnt[X] < 1: return None
        if ctx.cnt[X] == 1:
            lam = int(ctx.lamx[X])
        elif rng is not None:
            L = ctx.lams(X)
            if not L: return None
            lam = rng.choice(L)
        else:
            continue
        row = 0
        for b in range(k):
            if lam >> b & 1: row ^= (1 << (i * k + b)) ^ (1 << (j * k + b))
        rows.append(row)
    return rows

def grow(ctx, rng, order=None, limit=None):
    G = order if order is not None else random.Random(rng.randrange(1 << 30)).sample(ctx.good, len(ctx.good))
    F = []; el = Elim()
    for S in G:
        rows = pair_rows(ctx, F, S)
        if rows is None: continue
        tr = el.copy(); ok = True
        for r in rows:
            if not tr.add(r, 1): ok = False; break
        if ok:
            el = tr; F.append(S)
            if limit and len(F) >= limit: break
    return F, el

def solve_signs(ctx, F, el, rng, tries=300):
    """find syndromes satisfying every constraint (incl. the disjunctive ones)"""
    k = ctx.k; nv = len(F) * k
    hard = []
    for i in range(len(F)):
        for j in range(i + 1, len(F)):
            X = F[i] & F[j]; t = ctx.PCT[X]
            if t <= 4: continue
            L = ctx.lams(X)
            if not L: return None
            if len(L) > 1: hard.append((i, j, L))
    for _ in range(tries):
        y = el.sample(nv, rng)
        s = [sum(y[i * k + b] << b for b in range(k)) for i in range(len(F))]
        if all(any(par(l & (s[i] ^ s[j])) for l in L) for i, j, L in hard):
            return [ctx.rep[x] for x in s]
    return None

def build(F, V, c, n):
    out = []
    for S, cs in zip(F, c):
        idx = [b for b in range(n) if S >> b & 1]
        for w in {(cs ^ v) & S for v in V}:
            vec = [0] * n
            for b in idx: vec[b] = -1 if (w >> b) & 1 else 1
            out.append(vec)
    return out

def check(W, n):
    A = np.array(W, dtype=np.int64)
    if A.shape[0] == 0: return False, 99
    G = A @ A.T; np.fill_diagonal(G, -99)
    ok = (G.max() <= 4 and len(set(map(tuple, W))) == len(W)
          and (A * A).sum(1).min() == 8 and (A * A).sum(1).max() == 8)
    return ok, int(G.max())
