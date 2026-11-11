/* Max clique on an arbitrary graph read from a file:
 *   first line: nv ne
 *   then ne lines: i j
 * ./gclique file seconds seed   -> prints clique size then the vertices */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef unsigned long long u64;
static int M,W; static u64 *A;
static u64 rs; static inline unsigned rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (unsigned)(rs>>32); }
static inline int getb(const u64*r,int j){ return (r[j>>6]>>(j&63))&1; }
int main(int argc,char**argv){
    FILE*f=fopen(argv[1],"r"); double secs=atof(argv[2]);
    rs=strtoull(argv[3],0,10)*88172645463325252ULL+12345; for(int i=0;i<40;i++) rnd();
    long ne; if(fscanf(f,"%d %ld",&M,&ne)!=2) return 1;
    W=(M+63)/64; A=calloc((size_t)M*W,8);
    for(long e=0;e<ne;e++){ int i,j; if(fscanf(f,"%d %d",&i,&j)!=2) break;
        A[(size_t)i*W+(j>>6)]|=1ULL<<(j&63); A[(size_t)j*W+(i>>6)]|=1ULL<<(i&63); }
    fclose(f);
    int *cur=malloc(sizeof(int)*M),*best=malloc(sizeof(int)*M),nbest=0,ncur;
    u64 *cand=malloc(8*W),*tmp=malloc(8*W);
    int *pool=malloc(sizeof(int)*M);
    clock_t t0=clock();
    while((double)(clock()-t0)/CLOCKS_PER_SEC<secs){
        ncur=0; memset(cand,0,8*W);
        for(int j=0;j<M;j++) cand[j>>6]|=1ULL<<(j&63);
        while(1){
            int np=0;
            for(int w=0;w<W;w++){ u64 x=cand[w]; while(x){ int b=__builtin_ctzll(x); x&=x-1; pool[np++]=(w<<6)|b; } }
            if(!np) break;
            int v;
            if(rnd()%100<55) v=pool[rnd()%np];
            else { int bd=-1; v=pool[0];
                for(int q=0;q<np;q++){ int u=pool[q],c=0; const u64*r=A+(size_t)u*W;
                    for(int w=0;w<W;w++) c+=__builtin_popcountll(r[w]&cand[w]);
                    if(c>bd){bd=c;v=u;} } }
            cur[ncur++]=v;
            const u64*r=A+(size_t)v*W; for(int w=0;w<W;w++) cand[w]&=r[w];
        }
        if(ncur>nbest){ nbest=ncur; memcpy(best,cur,sizeof(int)*ncur); }
    }
    printf("%d\n",nbest);
    for(int i=0;i<nbest;i++) printf("%d ",best[i]); printf("\n");
    return 0;
}
