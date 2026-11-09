/* Maximum independent set over weight-8 sign vectors in Z^n (n = 13 or 14).
 *
 * Candidates: v in Z^n with exactly 8 nonzero entries, all +-1  (norm^2 = 8).
 * Constraint: <u,v> <= 4 for u != v.  With |S ^ T| = t and sign masks s,u
 * (bitmasks over the n coordinates, supported on S resp. T),
 *      <u,v> = t - 2*popcount((s^u) & S & T),
 * so the pair CONFLICTS iff popcount((s^u)&S&T) < req(t), where
 *      req(t) = 0 for t<=4,  1 for t in {5,6},  2 for t in {7,8}.
 *
 * All 4*C(n,2) vectors of type (+-2,+-2,0^(n-2)) are compatible with each
 * other and with every weight-8 vector, so the kissing count for this model
 * is 4*C(n,2) + |independent set|.
 *
 *   ./mis8 n seconds seed [seedfile]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned int u32; typedef unsigned long long u64;
static int n, NS;                 /* #supports = C(n,8) */
static int *supmask;              /* support index -> n-bit mask */
static int *idxof;                /* n-bit mask -> support index (or -1) */
static unsigned char *cmp;        /* [support][mask] -> 8-bit compressed sign */
static int *nbrT, *nbrC; static unsigned char *nbrR; /* per support: partner list */
static int *nbrOff, *nbrCnt;
static int NV;                    /* NS * 256 */
static unsigned short *tight; static unsigned char *inX;
static int *sol, nsol;            /* solution list */
static int *pos;                  /* vertex -> index in sol, or -1 */

static u64 rs;
static inline u32 rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (u32)(rs>>32); }
static inline int pc(int x){ return __builtin_popcount(x); }

/* vertex id = support index * 256 + compressed 8-bit sign pattern */
static inline int expand(int si,int cs){           /* 8-bit -> n-bit mask */
    int m=supmask[si], out=0, k=0;
    for(int b=0;b<n;b++) if(m>>b&1){ if(cs>>k&1) out|=1<<b; k++; }
    return out;
}
static int *expTab;                                 /* [si*256+cs] -> n-bit mask */

static inline int req_of(int t){ return t<=4?0 : (t<=6?1:2); }

/* enumerate neighbours of vertex v, calling body(w) */
#define FOR_NBRS(v, W, body) do{                                            \
    int _si=(v)>>8, _cs=(v)&255, _sm=expTab[(v)];                           \
    for(int _e=nbrOff[_si]; _e<nbrOff[_si]+nbrCnt[_si]; _e++){              \
        int _tj=nbrT[_e], _cm=nbrC[_e], _rq=nbrR[_e];                       \
        int _base=_sm&_cm, _free=supmask[_tj]&~_cm;                         \
        if(_rq==1){                                                         \
            int _f=_free, _sub=0;                                           \
            do{ int _um=_base|_sub; int W=(_tj<<8)|cmp[(size_t)_tj*(1<<14)+_um]; \
                body; _sub=(_sub-_f)&_f; }while(_sub);                      \
        } else {                                                            \
            /* t=7: flip 0 or 1 of the 7 common bits, 1 free bit */         \
            int _f=_free, _sub=0;                                           \
            do{ int _c=_cm, _bit=0;                                          \
                for(int _q=0;_q<8;_q++){                                     \
                    int _um;                                                 \
                    if(_q==0) _um=_base|_sub;                                \
                    else { if(!_c) break; _bit=_c&-_c; _c^=_bit;             \
                           _um=(_base^_bit)|_sub; }                          \
                    { int W=(_tj<<8)|cmp[(size_t)_tj*(1<<14)+_um]; body; }   \
                }                                                            \
                _sub=(_sub-_f)&_f; }while(_sub);                             \
        }                                                                   \
    }                                                                       \
    /* same support: flip exactly one of the 8 bits */                      \
    for(int _k=0;_k<8;_k++){ int W=(_si<<8)|(_cs^(1<<_k)); body; }          \
}while(0)

static inline int adj(int a,int b){
    if(a==b) return 0;
    int sa=a>>8, sb=b>>8, cm=supmask[sa]&supmask[sb], t=pc(cm);
    if(t<=4) return 0;
    return pc((expTab[a]^expTab[b])&cm) < req_of(t);
}

static void addv(int v){ inX[v]=1; pos[v]=nsol; sol[nsol++]=v;
    FOR_NBRS(v,W,{ tight[W]++; }); }
static void delv(int v){ inX[v]=0; int p=pos[v]; sol[p]=sol[--nsol]; pos[sol[p]]=p; pos[v]=-1;
    FOR_NBRS(v,W,{ tight[W]--; }); }

int main(int argc,char**argv){
    n=atoi(argv[1]); double secs=atof(argv[2]); rs=strtoull(argv[3],0,10)*6364136223846793005ULL+1442695040888963407ULL;
    for(int i=0;i<50;i++) rnd();
    /* supports */
    idxof=malloc(sizeof(int)<<14); for(int i=0;i<(1<<14);i++) idxof[i]=-1;
    NS=0; supmask=malloc(sizeof(int)*10000);
    for(int m=0;m<(1<<n);m++) if(pc(m)==8){ idxof[m]=NS; supmask[NS++]=m; }
    NV=NS*256;
    fprintf(stderr,"n=%d supports=%d vertices=%d\n",n,NS,NV);
    cmp=malloc((size_t)NS<<14);
    expTab=malloc(sizeof(int)*(size_t)NV);
    for(int si=0;si<NS;si++){
        for(int cs=0;cs<256;cs++){ int em=expand(si,cs); expTab[si*256+cs]=em; }
        int m=supmask[si];
        for(int sub=m;;sub=(sub-1)&m){ int k=0,c=0;
            for(int b=0;b<n;b++) if(m>>b&1){ if(sub>>b&1) c|=1<<k; k++; }
            cmp[((size_t)si<<14)|sub]=c; if(!sub) break; }
    }
    /* neighbour support lists */
    nbrOff=malloc(sizeof(int)*NS); nbrCnt=calloc(NS,sizeof(int));
    long cap=0; for(int si=0;si<NS;si++){ int c=0;
        for(int sj=0;sj<NS;sj++) if(sj!=si && pc(supmask[si]&supmask[sj])>=5) c++;
        nbrCnt[si]=c; nbrOff[si]=cap; cap+=c; }
    nbrT=malloc(sizeof(int)*cap); nbrC=malloc(sizeof(int)*cap); nbrR=malloc(cap);
    for(int si=0;si<NS;si++){ long e=nbrOff[si];
        for(int sj=0;sj<NS;sj++) if(sj!=si){ int cm=supmask[si]&supmask[sj],t=pc(cm);
            if(t>=5){ nbrT[e]=sj; nbrC[e]=cm; nbrR[e]=req_of(t); e++; } } }
    fprintf(stderr,"support-adjacency entries: %ld\n",cap);
    tight=calloc(NV,sizeof(unsigned short)); inX=calloc(NV,1);
    sol=malloc(sizeof(int)*NV); pos=malloc(sizeof(int)*NV);
    for(int i=0;i<NV;i++) pos[i]=-1;
    nsol=0;
    if(argc>4){ FILE*f=fopen(argv[4],"r"); int a,b;
        while(fscanf(f,"%d %d",&a,&b)==2){ int v=a*256+b; if(!tight[v]&&!inX[v]) addv(v); }
        fclose(f); fprintf(stderr,"seeded solution size %d\n",nsol); }
    int best=nsol; int *bestsol=malloc(sizeof(int)*NV); memcpy(bestsol,sol,sizeof(int)*nsol);
    int *free_=malloc(sizeof(int)*NV);
    clock_t t0=clock(); long iter=0;
    int *cand=malloc(sizeof(int)*8192);
    while((double)(clock()-t0)/CLOCKS_PER_SEC < secs){
        iter++;
        /* greedily insert free vertices in random order */
        int nf=0; int start=rnd()%NV;
        for(int i=0;i<NV;i++){ int v=start+i; if(v>=NV) v-=NV;
            if(!inX[v] && !tight[v]) free_[nf++]=v; }
        for(int i=0;i<nf;i++){ int v=free_[i]; if(!inX[v] && !tight[v]) addv(v); }
        if(nsol>best){ best=nsol; memcpy(bestsol,sol,sizeof(int)*nsol);
            fprintf(stderr,"iter %ld best %d (t=%.0fs)\n",iter,best,(double)(clock()-t0)/CLOCKS_PER_SEC); }
        /* (1,2)-swaps */
        int improved=1, rounds=0;
        while(improved && rounds++<60){ improved=0;
            for(int p=0;p<nsol;p++){ int v=sol[p]; int nc=0;
                FOR_NBRS(v,W,{ if(tight[W]==1 && !inX[W] && nc<8000) cand[nc++]=W; });
                for(int a=0;a<nc && !improved;a++) for(int b=a+1;b<nc;b++)
                    if(cand[a]!=cand[b] && !adj(cand[a],cand[b])){
                        int x=cand[a],y=cand[b]; delv(v); addv(x); addv(y); improved=1; break; }
                if(improved) break; }
            if(improved && nsol>best){ best=nsol; memcpy(bestsol,sol,sizeof(int)*nsol);
                fprintf(stderr,"swap best %d (t=%.0fs)\n",best,(double)(clock()-t0)/CLOCKS_PER_SEC); }
        }
        /* perturb: force in a few random vertices */
        int kick=1+rnd()%4;
        for(int q=0;q<kick;q++){ int v=rnd()%NV; if(inX[v]) continue;
            FOR_NBRS(v,W,{ if(inX[W]) delv(W); });
            if(!tight[v]) addv(v); }
    }
    printf("n=%d best_weight8=%d  total_with_Droots=%d\n",n,best,best+4*n*(n-1)/2);
    char fn[128]; snprintf(fn,sizeof fn,"lower_bound/logs/mis8_n%d_s%s.txt",n,argv[3]);
    FILE*f=fopen(fn,"w");
    for(int i=0;i<best;i++) fprintf(f,"%d %d\n",bestsol[i]>>8,bestsol[i]&255);
    fclose(f); fprintf(stderr,"wrote %s (%d vertices)\n",fn,best);
    return 0;
}
