"""dim-14: search for a global linear code V <= F_2^14 admitting MORE than
Ganzhinov's 49 supports.  m supports x 2^dim(V) weight-8 vectors + 364 D-roots;
beating 1932 needs m*2^d >= 1569 (d=5 -> m >= 50, d=6 -> m >= 25)."""
import sys, os, json, random, time
sys.path.insert(0, 'lower_bound/lib')
import numpy as np
from gcode import Ctx, Elim, pair_rows, solve_signs, build, check, code_words
from pipeline import structural_graph, max_clique

def random_code_minwt(n, d, wmin, rng, tries=400):
    """random linear code of dimension d in F_2^n with minimum weight >= wmin,
    built greedily one basis vector at a time (rejection on the whole span)."""
    for _ in range(tries):
        basis = []; words = [0]
        ok = True
        for _ in range(d):
            cand = [x for x in range(1, 1 << n)
                    if all(bin(x ^ w).count('1') >= wmin for w in words)]
            if not cand: ok = False; break
            b = rng.choice(cand); basis.append(b)
            words = words + [w ^ b for w in words]
        if ok and len(set(words)) == 1 << d: return words
    return None

n = 14; DROOTS = 364; RECORD = 1932
PCT = np.array([bin(x).count('1') for x in range(1 << n)], dtype=np.int8)
SUPS = np.array([m for m in range(1 << n) if bin(m).count('1') == 8], dtype=np.int32)
OUT = 'lower_bound/dim14/configs'; os.makedirs(OUT, exist_ok=True)

def consistent(ctx, F, rng, tries=25):
    best = []
    for _ in range(tries):
        order = F[:]; rng.shuffle(order)
        keep = []; el = Elim()
        for S in order:
            rows = pair_rows(ctx, keep, S, rng)
            if rows is None: continue
            tr = el.copy()
            if all(tr.add(r, 1) for r in rows): el = tr; keep.append(S)
        if len(keep) > len(best): best, bel = keep, el
    return best, bel

def run(seed, budget):
    rng = random.Random(seed); best = 0; t0 = time.time(); tried = 0
    while time.time() - t0 < budget:
        # Griesmer caps the minimum weight of a [14,5] or [14,6] code at 6, and
        # Ganzhinov's V attains it.  Low-weight words are exactly what breaks
        # 'goodness' (every nonzero codeword must meet every 8-support twice),
        # so sample only minimum-weight-6 codes.
        d = 5; tried += 1        # [14,6,6] codes are not reachable by the greedy builder
        V = random_code_minwt(n, d, 6, rng)
        if V is None: continue
        ctx = Ctx(V, n, PCT, SUPS)
        if len(ctx.good) < 60: continue
        G, edges = structural_graph(ctx, cap=3100)
        if not edges: continue
        cl = max_clique(len(G), edges, 8.0, rng.randrange(1 << 30))
        if len(cl) * (1 << d) + DROOTS < 900: continue
        print(f'  V minwt6: good={len(ctx.good)} clique={len(cl)}', flush=True)
        F = [G[i] for i in cl]
        keep, el = consistent(ctx, F, rng)
        if len(keep) * (1 << d) + DROOTS <= best: continue
        c = None
        for _ in range(200):                      # every constraint is now linear
            y = el.sample(len(keep) * ctx.k, rng)
            cand = [ctx.rep[sum(y[i * ctx.k + b] << b for b in range(ctx.k))]
                    for i in range(len(keep))]
            W0 = build(keep, V, cand, n)
            if check(W0, n)[0]: c = cand; break
        if c is None: continue
        W = build(keep, V, c, n)
        ok, mx = check(W, n)
        if not ok: continue
        tot = len(W) + DROOTS
        if tot > best:
            best = tot
            msg = (f'seed={seed} dimV={d} supports={len(keep)} weight8={len(W)} '
                   f'total={tot} ({"BEATS" if tot > RECORD else "below"} 1932)')
            print(msg, flush=True)
            with open('lower_bound/logs/progress.log', 'a') as f:
                f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' | 14 | gcode-search | ' + msg + '\n')
            if tot > RECORD:
                json.dump({'V': V, 'supports': keep, 'cosets': c, 'dim_V': d,
                           'weight8': len(W), 'total': tot, 'n': n},
                          open(f'{OUT}/gc_s{seed}_{tot}.json', 'w'))
    print(f'seed {seed}: {tried} codes, best {best}', flush=True)

if __name__ == '__main__':
    run(int(sys.argv[1]), float(sys.argv[2]))
