/* Given a family of 8-element supports in [n], find the largest set of
 * (support, sign) pairs with all pairwise inner products <= 4.
 * Conflict rule:  popcount((s^u) & S_i & S_j) < req(t),  t=|S_i^S_j|,
 *                 req = 0 (t<=4), 1 (t=5,6), 2 (t=7,8).
 * Dense-bitset iterated local search (greedy + (1,2)-swaps + perturbation).
 *   ./signmis suppfile n seconds seed [startfile]
 * suppfile: one integer bitmask per line. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef unsigned long long u64;
static int m, NV, W, n;
static int *S;            /* support masks */
static int *expT;         /* vertex -> n-bit sign mask (subset of its support) */
static u64 *A;            /* adjacency bitset */
static unsigned short *tight; static unsigned char *inX;
static int *sol,nsol,*pos;
static u64 rs; static inline unsigned rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (unsigned)(rs>>32); }
static inline int pc(int x){ return __builtin_popcount(x); }
static inline int req_of(int t){ return t<=4?0:(t<=6?1:2); }
static inline int getb(const u64*r,int j){ return (r[j>>6]>>(j&63))&1; }
static void addv(int v){ inX[v]=1; pos[v]=nsol; sol[nsol++]=v;
    const u64*r=A+(size_t)v*W; for(int w=0;w<W;w++){ u64 x=r[w]; while(x){ int b=__builtin_ctzll(x); x&=x-1; tight[(w<<6)|b]++; } } }
static void delv(int v){ inX[v]=0; int p=pos[v]; sol[p]=sol[--nsol]; pos[sol[p]]=p; pos[v]=-1;
    const u64*r=A+(size_t)v*W; for(int w=0;w<W;w++){ u64 x=r[w]; while(x){ int b=__builtin_ctzll(x); x&=x-1; tight[(w<<6)|b]--; } } }
int main(int argc,char**argv){
    FILE*f=fopen(argv[1],"r"); n=atoi(argv[2]); double secs=atof(argv[3]);
    rs=strtoull(argv[4],0,10)*6364136223846793005ULL+1442695040888963407ULL;
    for(int i=0;i<40;i++) rnd();
    S=malloc(sizeof(int)*4096); m=0; while(fscanf(f,"%d",&S[m])==1) m++; fclose(f);
    NV=m*256; W=(NV+63)/64;
    fprintf(stderr,"supports=%d vertices=%d adj=%.1fMB\n",m,NV,(double)NV*W*8/1e6);
    expT=malloc(sizeof(int)*NV);
    for(int i=0;i<m;i++){ int cs[8],k=0; for(int b=0;b<n;b++) if(S[i]>>b&1) cs[k++]=b;
        for(int t=0;t<256;t++){ int e=0; for(int j=0;j<8;j++) if(t>>j&1) e|=1<<cs[j]; expT[i*256+t]=e; } }
    A=calloc((size_t)NV*W,8);
    for(int i=0;i<m;i++) for(int j=0;j<m;j++){
        int cm=S[i]&S[j], t=pc(cm), rq=(i==j)?2:req_of(t); if(!rq) continue;
        for(int a=0;a<256;a++){ int va=i*256+a; const int ea=expT[va];
            for(int b=0;b<256;b++){ int vb=j*256+b; if(va==vb) continue;
                if(pc((ea^expT[vb])&cm)<rq) A[(size_t)va*W+(vb>>6)]|=1ULL<<(vb&63); } } }
    tight=calloc(NV,2); inX=calloc(NV,1); sol=malloc(sizeof(int)*NV); pos=malloc(sizeof(int)*NV);
    for(int i=0;i<NV;i++) pos[i]=-1;
    nsol=0;
    if(argc>5){ FILE*g=fopen(argv[5],"r"); int a,b; while(fscanf(g,"%d %d",&a,&b)==2){ int v=a*256+b;
        if(v<NV && !inX[v] && !tight[v]) addv(v);} fclose(g); fprintf(stderr,"seeded %d\n",nsol); }
    int best=nsol,*bs=malloc(sizeof(int)*NV); memcpy(bs,sol,sizeof(int)*nsol);
    clock_t t0=clock(); int cand[4096];
    while((double)(clock()-t0)/CLOCKS_PER_SEC<secs){
        int start=rnd()%NV;
        for(int i=0;i<NV;i++){ int v=start+i; if(v>=NV)v-=NV; if(!inX[v]&&!tight[v]) addv(v); }
        if(nsol>best){best=nsol;memcpy(bs,sol,sizeof(int)*nsol);
            fprintf(stderr,"best %d (%.0fs)\n",best,(double)(clock()-t0)/CLOCKS_PER_SEC);}
        int improved=1,rounds=0;
        while(improved&&rounds++<200){ improved=0;
            for(int p=0;p<nsol && !improved;p++){ int v=sol[p],nc=0; const u64*r=A+(size_t)v*W;
                for(int w=0;w<W&&nc<4000;w++){ u64 x=r[w]; while(x){ int b=__builtin_ctzll(x); x&=x-1;
                    int u=(w<<6)|b; if(!inX[u]&&tight[u]==1) cand[nc++]=u; if(nc>=4000) break; } }
                for(int a=0;a<nc&&!improved;a++) for(int b=a+1;b<nc;b++)
                    if(!getb(A+(size_t)cand[a]*W,cand[b])){ int x=cand[a],y=cand[b];
                        delv(v); addv(x); addv(y); improved=1; break; } }
            if(improved&&nsol>best){best=nsol;memcpy(bs,sol,sizeof(int)*nsol);
                fprintf(stderr,"swap best %d (%.0fs)\n",best,(double)(clock()-t0)/CLOCKS_PER_SEC);} }
        int kick=1+rnd()%6;
        for(int q=0;q<kick;q++){ int v=rnd()%NV; if(inX[v])continue; const u64*r=A+(size_t)v*W;
            for(int w=0;w<W;w++){ u64 x=r[w]; while(x){ int b=__builtin_ctzll(x); x&=x-1; int u=(w<<6)|b; if(inX[u]) delv(u);} }
            if(!tight[v]) addv(v); }
    }
    printf("supports=%d best_weight8=%d total_with_Droots=%d\n",m,best,best+4*n*(n-1)/2);
    char fn[256]; snprintf(fn,sizeof fn,"%s.sol%s",argv[1],argv[4]);
    FILE*g=fopen(fn,"w"); for(int i=0;i<best;i++) fprintf(g,"%d %d\n",bs[i]>>8,bs[i]&255); fclose(g);
    fprintf(stderr,"wrote %s\n",fn);
    return 0;
}
