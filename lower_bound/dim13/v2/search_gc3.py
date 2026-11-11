"""dim-13 global-linear-code search (fast).  Grows support families under the
exact F_2 coset-feasibility constraint; every hit is Gram-checked and saved."""
import sys, json, random, time
sys.path.insert(0, 'lower_bound/lib')
import numpy as np
from gcfast import CodeCtx, popcount_table
import cosets2
from cosets import build_vectors

N = 13; DROOTS = 312; RECORD = 1154
OUT = 'lower_bound/dim13/v2/configs'; LOG = 'lower_bound/logs/progress.log'
PCT = popcount_table(N)
SUPS = np.array([m for m in range(1 << N) if bin(m).count('1') == 8], dtype=np.int32)

def log(tag, msg):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + f' | 13 | {tag} | ' + msg + '\n')

def code_words(basis):
    out = [0]
    for b in basis: out += [x ^ b for x in out]
    return out

def grow(ctx, rng, order=None):
    cnt, lam, k = ctx.cnt, ctx.lam, ctx.k
    G = ctx.good.tolist()
    if order is None: order = G[:]; rng.shuffle(order)
    F = []; piv = {}
    def addrow(pv, row, rhs):
        while row:
            h = row.bit_length() - 1
            if h in pv: pr, prhs = pv[h]; row ^= pr; rhs ^= prhs
            else: pv[h] = (row, rhs); return True
        return rhs == 0
    for S in order:
        rows = []; ok = True
        for idx, T in enumerate(F):
            X = S & T; t = PCT[X]
            if t <= 4: continue
            if t >= 7 or cnt[X] < 1: ok = False; break
            if cnt[X] == 1:
                l = int(lam[X]); row = 0; j = len(F)
                for b in range(k):
                    if l >> b & 1: row ^= (1 << (idx * k + b)) ^ (1 << (j * k + b))
                rows.append(row)
        if not ok: continue
        trial = dict(piv); good = True
        for row in rows:
            if not addrow(trial, row, 1): good = False; break
        if good: piv = trial; F.append(S)
    return F

def evaluate(F, V, seed):
    c = cosets2.solve(F, V, N, seed=seed, restarts=30, sweeps=2500)
    if c is None: return None
    W = build_vectors(F, V, c, N)
    A = np.array(W, dtype=np.int64); Gm = A @ A.T; np.fill_diagonal(Gm, -99)
    if Gm.max() > 4: return None
    if len(set(map(tuple, W))) != len(W): return None
    if (A * A).sum(1).min() != 8 or (A * A).sum(1).max() != 8: return None
    return W, c

def run(seed, budget, dims=(4, 5, 6)):
    rng = random.Random(seed); best = 0; t0 = time.time(); tried = 0
    while time.time() - t0 < budget:
        d = rng.choice(dims); tried += 1
        basis = [rng.randrange(1, 1 << N) for _ in range(d)]
        V = code_words(basis)
        if len(set(V)) != 1 << d: continue
        ctx = CodeCtx(V, N, PCT, SUPS)
        if len(ctx.good) * (1 << d) + DROOTS < 800: continue
        for _rep in range(8):
            F = grow(ctx, rng)
            tot_guess = len(F) * (1 << d) + DROOTS
            if tot_guess <= best: continue
            r = evaluate(F, V, rng.randrange(10 ** 9))
            if r is None: continue
            W, c = r; tot = len(W) + DROOTS
            if tot > best:
                best = tot
                print(f'seed{seed} d={d} sup={len(F)} w8={len(W)} TOTAL={tot}', flush=True)
                log('gc3', f'seed={seed} dimV={d} supports={len(F)} weight8={len(W)} '
                          f'total={tot} ({"BEATS" if tot > RECORD else "below"} 1154)')
                json.dump({'V': V, 'supports': F, 'cosets': c, 'dim_V': d,
                           'weight8': len(W), 'total': tot, 'n': N},
                          open(f'{OUT}/gc3_s{seed}_{tot}.json', 'w'))
    print(f'seed {seed}: {tried} codes tried, best {best}', flush=True)
    return best

if __name__ == '__main__':
    run(int(sys.argv[1]), float(sys.argv[2]))
