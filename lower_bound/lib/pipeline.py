"""Full pipeline: code V -> good supports -> structural graph -> max clique
   -> F_2 coset consistency -> explicit vectors -> exact Gram check."""
import os, subprocess, tempfile, random
import numpy as np
from gcode import Ctx, Elim, pair_rows, solve_signs, build, check, code_words, pc

CLQ = os.path.abspath('lower_bound/lib/gclique')

def structural_graph(ctx, cap=4000):
    """edges = compatible support pairs; cap only guards against huge graphs"""
    G = ctx.good
    if len(G) > cap:
        G = random.sample(G, cap)
    Ga = np.array(G, dtype=np.int32)
    PCT = ctx.PCT
    edges = []
    for i in range(len(Ga)):
        X = np.bitwise_and(Ga[i], Ga[i + 1:])
        t = PCT[X]
        ok = (t <= 4) | (((t == 5) | (t == 6)) & (ctx.cnt[X] >= 1))
        for j in np.nonzero(ok)[0]:
            edges.append((i, i + 1 + int(j)))
    return G, edges

def max_clique(nv, edges, secs=2.0, seed=1):
    with tempfile.NamedTemporaryFile('w', suffix='.g', delete=False) as f:
        f.write(f"{nv} {len(edges)}\n")
        f.write('\n'.join(f"{a} {b}" for a, b in edges))
        path = f.name
    out = subprocess.run([CLQ, path, str(secs), str(seed)], capture_output=True, text=True).stdout.split('\n')
    os.unlink(path)
    return [int(x) for x in out[1].split()] if len(out) > 1 and out[1].strip() else []

def consistent_subfamily(ctx, F, rng):
    """keep the largest prefix-greedy subfamily whose forced system is consistent"""
    order = F[:]; rng.shuffle(order)
    keep = []; el = Elim()
    for S in order:
        rows = pair_rows(ctx, keep, S)
        if rows is None: continue
        tr = el.copy()
        if all(tr.add(r, 1) for r in rows): el = tr; keep.append(S)
    return keep, el

def attempt(V, n, PCT, SUPS, rng, clique_secs=2.0, tries=200):
    ctx = Ctx(V, n, PCT, SUPS)
    if len(ctx.good) < 4: return None
    G, edges = structural_graph(ctx)
    cl = max_clique(len(G), edges, clique_secs, rng.randrange(1 << 30))
    if not cl: return None
    F = [G[i] for i in cl]
    keep, el = consistent_subfamily(ctx, F, rng)
    if not keep: return None
    c = solve_signs(ctx, keep, el, rng, tries=tries)
    if c is None: return None
    W = build(keep, V, c, n)
    ok, mx = check(W, n)
    if not ok: return None
    return keep, c, W, len(ctx.good), len(F)
