/* Max clique on k-subsets of [n] whose pairwise intersections lie in an
 * allowed set.  ./clique n k allowedmask seconds seed
 * allowedmask: bit i set  <=>  intersection size i is allowed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef unsigned long long u64;
static int M; static u64 *A; static int W; /* adjacency bitset rows */
static int *msk;
static u64 rs; static inline unsigned rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (unsigned)(rs>>32); }
static inline int pc(int x){ return __builtin_popcount(x); }
static inline int getb(u64*r,int j){ return (r[j>>6]>>(j&63))&1; }
int main(int argc,char**argv){
    int n=atoi(argv[1]),k=atoi(argv[2]); unsigned allowed=strtoul(argv[3],0,0);
    double secs=atof(argv[4]); rs=strtoull(argv[5],0,10)*88172645463325252ULL+12345;
    for(int i=0;i<40;i++) rnd();
    int cap=1<<n; msk=malloc(sizeof(int)*cap); M=0;
    for(int m=0;m<cap;m++) if(pc(m)==k) msk[M++]=m;
    W=(M+63)/64; A=calloc((size_t)M*W,8);
    for(int i=0;i<M;i++) for(int j=0;j<M;j++) if(i!=j){
        int t=pc(msk[i]&msk[j]); if(allowed>>t&1) A[(size_t)i*W+(j>>6)]|=1ULL<<(j&63); }
    fprintf(stderr,"vertices=%d\n",M);
    int *cur=malloc(sizeof(int)*M),ncur,*best=malloc(sizeof(int)*M),nbest=0;
    u64 *cand=malloc(8*W);
    clock_t t0=clock();
    while((double)(clock()-t0)/CLOCKS_PER_SEC<secs){
        /* randomised greedy from a random seed vertex, then plateau restarts */
        ncur=0; memset(cand,0xFF,8*W);
        for(int j=M;j<W*64;j++) cand[j>>6]&=~(1ULL<<(j&63));
        while(1){
            int cnt=0; for(int w=0;w<W;w++) cnt+=__builtin_popcountll(cand[w]);
            if(!cnt) break;
            int pick=rnd()%cnt, v=-1;
            for(int j=0;j<M && v<0;j++) if(getb(cand,j)){ if(pick--==0) v=j; }
            cur[ncur++]=v;
            for(int w=0;w<W;w++) cand[w]&=A[(size_t)v*W+w];
        }
        if(ncur>nbest){ nbest=ncur; memcpy(best,cur,sizeof(int)*ncur);
            fprintf(stderr,"clique %d (%.0fs)\n",nbest,(double)(clock()-t0)/CLOCKS_PER_SEC); }
    }
    printf("max_clique_found=%d\n",nbest);
    for(int i=0;i<nbest;i++){ printf("%d:",msk[best[i]]);
        for(int b=0;b<n;b++) if(msk[best[i]]>>b&1) printf(" %d",b); printf("\n"); }
    return 0;
}
