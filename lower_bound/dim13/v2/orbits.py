"""Highly-symmetric (group-orbit) configurations in R^13.

A subgroup G <= S_14 acts on the zero-sum hyperplane of R^14, a 13-dimensional
orthogonal representation.  For x in that hyperplane, the orbit G.x is a
kissing configuration iff  max_{g: gx != x} <x, gx> <= |x|^2 / 2.
Optimises x to minimise that maximum, over many random starts, for several
groups (PSL(2,13), PGL(2,13), and subgroups)."""
import sys, itertools, random, numpy as np

def psl2_13():
    q = 13; pts = list(range(q)) + ['inf']
    idx = {p: i for i, p in enumerate(pts)}
    def act(a, b, c, d, x):
        if x == 'inf': return idx[(a * pow(c, -1, q)) % q] if c % q else idx['inf']
        num = (a * x + b) % q; den = (c * x + d) % q
        return idx['inf'] if den == 0 else idx[num * pow(den, -1, q) % q]
    sq = {(i * i) % q for i in range(1, q)}
    perms = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    det = (a * d - b * c) % q
                    if det == 0 or det not in sq: continue
                    perms.add(tuple(act(a, b, c, d, x) for x in pts))
    return sorted(perms)

def pgl2_13():
    q = 13; pts = list(range(q)) + ['inf']
    idx = {p: i for i, p in enumerate(pts)}
    def act(a, b, c, d, x):
        if x == 'inf': return idx[(a * pow(c, -1, q)) % q] if c % q else idx['inf']
        num = (a * x + b) % q; den = (c * x + d) % q
        return idx['inf'] if den == 0 else idx[num * pow(den, -1, q) % q]
    perms = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    if (a * d - b * c) % q == 0: continue
                    perms.add(tuple(act(a, b, c, d, x) for x in pts))
    return sorted(perms)

def orbit_max(P, x):
    """P: (m,14) permutation index array; returns (max inner product/|x|^2, orbit size)"""
    Y = x[P]                                  # each row is g.x
    d = Y @ x
    n2 = x @ x
    same = np.all(np.abs(Y - x) < 1e-12, axis=1)
    if same.all(): return 1.0, 1
    return float(d[~same].max() / n2), int(len(np.unique(np.round(Y, 9), axis=0)))

def optimise(P, seed, iters=4000, n=14):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n); x -= x.mean(); x /= np.linalg.norm(x)
    best, _ = orbit_max(P, x); bx = x.copy(); step = 0.25
    for it in range(iters):
        y = bx + step * rng.standard_normal(n)
        y -= y.mean(); ny = np.linalg.norm(y)
        if ny < 1e-9: continue
        y /= ny
        v, _ = orbit_max(P, y)
        if v < best: best, bx = v, y; step *= 1.05
        else: step *= 0.995
        if step < 1e-6: step = 0.25
    return best, bx

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'psl'
    G = psl2_13() if which == 'psl' else pgl2_13()
    P = np.array(G, dtype=np.int64)
    print(f"{which}: |G| = {len(G)}")
    best = (9, None)
    for s in range(int(sys.argv[2]) if len(sys.argv) > 2 else 60):
        v, x = optimise(P, s)
        if v < best[0]:
            best = (v, x)
            sz = orbit_max(P, x)[1]
            print(f"  seed {s}: max cos = {v:.6f}  orbit size {sz}  "
                  f"{'FEASIBLE (<=1/2)' if v <= 0.5 + 1e-9 else ''}", flush=True)
    print(f"best max cos over orbits: {best[0]:.6f} (need <= 0.5); orbit size "
          f"{orbit_max(P,best[1])[1]}")
    np.save(f'/tmp/claude-0/-home-user-unsolved/153d4bbc-6a4f-55aa-a990-7da3460d88ca/scratchpad/orb_{which}.npy', best[1])
