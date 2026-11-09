import numpy as np, itertools, collections, json
w8 = np.load('lower_bound/lib/g14_w8.npy')
sup = collections.defaultdict(list)
for v in w8:
    s = tuple(np.nonzero(v)[0]); sup[s].append(tuple((v[list(s)]>0).astype(int)))
sups = sorted(sup)
print("supports:", len(sups))
# is each sign code linear (affine) over F2?
def tobits(t): 
    x=0
    for i,b in enumerate(t): x |= b<<i
    return x
lin=[]
for s in sups:
    C = sorted(tobits(t) for t in sup[s])
    c0 = C[0]; V = set(c^c0 for c in C)
    isl = all((a^b) in V for a in V for b in V)
    lin.append(isl)
print("all sign codes affine over F2:", all(lin), " sizes:", set(len(sup[s]) for s in sups))
# structure of one code
s0=sups[0]; C0=sorted(tobits(t) for t in sup[s0]); c0=C0[0]
V0=sorted(c^c0 for c in C0)
wt=lambda x: bin(x).count('1')
print("support0",s0,"linear code weight distribution:", collections.Counter(wt(x) for x in V0))
# design structure of the 49 supports
pts=list(range(14))
inc = np.zeros((49,14),dtype=int)
for i,s in enumerate(sups): inc[i,list(s)]=1
print("point degrees:", collections.Counter(inc.sum(0).tolist()))
# pair degrees
pd = inc.T@inc
print("pair codegrees:", collections.Counter(pd[np.triu_indices(14,1)].tolist()))
# complements are 6-sets
comps=[tuple(sorted(set(range(14))-set(s))) for s in sups]
ci=collections.Counter()
for a,b in itertools.combinations(comps,2): ci[len(set(a)&set(b))]+=1
print("complement(6-set) pair intersections:", dict(sorted(ci.items())))
# do the 49 6-sets form a nice design? check if partition into 7 groups of 7
cinc=np.zeros((49,14),dtype=int)
for i,c in enumerate(comps): cinc[i,list(c)]=1
print("6-set point degrees:", collections.Counter(cinc.sum(0).tolist()))
print("6-set pair codegrees:", collections.Counter((cinc.T@cinc)[np.triu_indices(14,1)].tolist()))
json.dump({'supports':[list(map(int,s)) for s in sups],
           'codes':[sorted(tobits(t) for t in sup[s]) for s in sups]}, open('lower_bound/lib/g14_struct.json','w'))
