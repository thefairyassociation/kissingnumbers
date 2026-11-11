"""Max families of 8-subsets of [13] with pairwise intersections in {4,6}
   (equivalently 5-subsets with pairwise intersections in {1,3}).
   These are exactly the 'Ganzhinov pattern' supports: t=4 pairs are free,
   t=6 pairs only need disjoint sign projections."""
import itertools, random, sys
n = 13
fives = [f for f in itertools.combinations(range(n), 5)]
mask = [sum(1 << i for i in f) for f in fives]
M = len(fives)
pc = lambda x: bin(x).count('1')
adj = [set() for _ in range(M)]
for i in range(M):
    for j in range(i + 1, M):
        t = pc(mask[i] & mask[j])
        if t in (1, 3):
            adj[i].add(j); adj[j].add(i)
print("5-subsets:", M, "avg degree:", sum(len(a) for a in adj) / M)

best = []
random.seed(0)
for trial in range(4000):
    order = list(range(M)); random.shuffle(order)
    cur = []; curset = set()
    cand = set(range(M))
    while cand:
        v = max(cand, key=lambda x: (len(adj[x] & cand), random.random())) if trial % 4 == 0 else random.choice(sorted(cand))
        cur.append(v); cand = (cand & adj[v]) - {v}
    if len(cur) > len(best):
        best = cur[:]; print("trial", trial, "clique size", len(best), flush=True)
print("BEST family size:", len(best))
sets = [fives[i] for i in best]
print("5-sets:", sets)
import collections
print("pair intersections:", collections.Counter(len(set(a) & set(b)) for a, b in itertools.combinations(sets, 2)))
print("8-supports:", [sorted(set(range(n)) - set(s)) for s in sets])
