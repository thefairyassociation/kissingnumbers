import numpy as np, itertools, collections, sys
V = np.loadtxt('lower_bound/dim14/configs/ganzhinov_1932.txt', delimiter=',', dtype=np.int64)
print("shape", V.shape)
n2 = (V*V).sum(1)
print("norm2 values", collections.Counter(n2.tolist()))
G = V @ V.T
np.fill_diagonal(G, -10**9)
print("max offdiag", G.max(), " (need <= 4)")
# classify
types = collections.Counter()
for v in V:
    types[tuple(sorted(collections.Counter(np.abs(v)).items()))] += 1
print("types:", types)
w8 = V[np.count_nonzero(V,axis=1)==8]
w2 = V[np.count_nonzero(V,axis=1)==2]
print("n w8", len(w8), "n w2", len(w2))
sup = collections.Counter()
for v in w8:
    sup[tuple(np.nonzero(v)[0])] += 1
print("distinct 8-supports used:", len(sup))
print("per-support count histogram:", collections.Counter(sup.values()))
sups = sorted(sup)
# pairwise support intersections
inter = collections.Counter()
for a,b in itertools.combinations(sups,2):
    inter[len(set(a)&set(b))]+=1
print("support pair intersection sizes:", dict(sorted(inter.items())))
# is the set antipodal?
S = set(map(tuple, V.tolist()))
print("antipodal:", all(tuple((-np.array(v)).tolist()) in S for v in V))
np.save('/home/user/unsolved/lower_bound/lib/g14_w8.npy', w8)
import json
json.dump([list(map(int,s)) for s in sups], open('/home/user/unsolved/lower_bound/lib/g14_supports.json','w'))
