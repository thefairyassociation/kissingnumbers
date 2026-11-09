"""Fast primitives for the global-linear-code framework (numpy, n <= 14)."""
import numpy as np

def popcount_table(n):
    t = np.zeros(1 << n, dtype=np.int8)
    for b in range(n): t[1 << b:] = t[1 << b:]  # placeholder
    t = np.array([bin(x).count('1') for x in range(1 << n)], dtype=np.int8)
    return t

class CodeCtx:
    """everything derived from a linear code V <= F_2^n"""
    def __init__(self, V, n, PCT, SUPS):
        self.n = n; self.V = V; self.d = len(V).bit_length() - 1
        self.PCT = PCT; self.SUPS = SUPS
        full = (1 << n) - 1
        nz = np.array([v for v in V if v], dtype=np.int32)
        self.nz = nz
        # good supports: every nonzero codeword meets S in >= 2 coordinates
        ok = np.ones(len(SUPS), dtype=bool)
        for v in nz:
            ok &= PCT[np.bitwise_and(SUPS, int(v))] >= 2
        self.good = SUPS[ok]
        # V^perp
        allv = np.arange(1 << n, dtype=np.int32)
        keep = np.ones(1 << n, dtype=bool)
        for v in nz:
            keep &= (PCT[np.bitwise_and(allv, int(v))] & 1) == 0
        self.perp = allv[keep]
        # basis of V^perp and the coordinate map a -> lambda
        B = []
        for a in self.perp.tolist():
            x = a
            for b in B: x = min(x, x ^ b)
            if x: B.append(x); B.sort(reverse=True)
        self.B = B; self.k = len(B)
        coord = np.zeros(1 << n, dtype=np.int32)
        for a in self.perp.tolist():
            x = a; lam = 0
            for idx, b in enumerate(B):
                if x ^ b < x: x ^= b; lam |= 1 << idx
            coord[a] = lam
        # zeta transforms:  cnt[X] = #{a in perp, a != 0, a subset of X}
        #                   lam[X] = XOR of coord[a] over the same set
        cnt = np.zeros(1 << n, dtype=np.int32)
        lam = np.zeros(1 << n, dtype=np.int32)
        for a in self.perp.tolist():
            if a: cnt[a] += 1; lam[a] ^= int(coord[a])
        for b in range(n):
            bit = 1 << b
            idx = np.arange(1 << n)
            hi = idx[(idx & bit) != 0]
            cnt[hi] += cnt[hi ^ bit]
            lam[hi] ^= lam[hi ^ bit]
        self.cnt = cnt; self.lam = lam
