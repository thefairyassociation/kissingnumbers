import json, itertools, collections
d=json.load(open('lower_bound/lib/g14_struct.json'))
sups=[sum(1<<i for i in s) for s in d['supports']]; codes=d['codes']; n=14
def bits(m):
    return [b for b in range(n) if m>>b&1]
# lift each 8-bit code word to a 14-bit minus-set mask
lift=[]
for S,C in zip(sups,codes):
    cs=bits(S); lift.append([sum(1<<cs[k] for k in range(8) if w>>k&1) for w in C])
# linear part U_i (as 14-bit masks supported on S_i)
U=[sorted({w^L[0] for w in L}) for L in lift]
print("dims:", collections.Counter(len(u).bit_length()-1 for u in U))
# V = {x in F_2^14 : x|S_i in U_i for all i}  -- solve linear system by basis intersection
def span(vs):
    B=[]
    for v in vs:
        for b in B: v=min(v,v^b)
        if v: B.append(v); B.sort(reverse=True)
    return B
def inspan(B,v):
    for b in B:
        if v^b<v: v^=b
    return v==0
# brute force over F_2^14 (16384) is fine
Ub=[span(u) for u in U]
V=[x for x in range(1<<n) if all(inspan(Ub[i], x & sups[i]) for i in range(len(sups)))]
print("global V size:", len(V), "dim:", len(V).bit_length()-1)
Vb=span(V)
proj=[len({x&sups[i] for x in V}) for i in range(len(sups))]
print("dim of V|S_i:", collections.Counter(p.bit_length()-1 for p in proj))
print("matches U_i:", all({x&sups[i] for x in V}=={u for u in U[i]} for i in range(len(sups))))
# coset reps
reps=[L[0] for L in lift]
print("sample reps:", [hex(r) for r in reps[:6]])
