/* Which weight-8 sign vectors in Z^n are compatible with a given set?
 * Reads a solution file of "support_index sign_index" lines (same encoding as
 * mis8) and reports how many of the 2^8*C(n,8) candidates conflict with none.
 *   ./addable n solfile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static inline int pc(int x){ return __builtin_popcount(x); }
int main(int argc,char**argv){
    int n=atoi(argv[1]);
    int NS=0,*sup=malloc(sizeof(int)*10000),*idx=malloc(sizeof(int)<<14);
    for(int i=0;i<(1<<14);i++) idx[i]=-1;
    for(int m=0;m<(1<<n);m++) if(pc(m)==8){ idx[m]=NS; sup[NS++]=m; }
    int NV=NS*256;
    int *exp=malloc(sizeof(int)*(size_t)NV);
    for(int si=0;si<NS;si++){ int cs[8],k=0,m=sup[si];
        for(int b=0;b<n;b++) if(m>>b&1) cs[k++]=b;
        for(int t=0;t<256;t++){ int e=0; for(int j=0;j<8;j++) if(t>>j&1) e|=1<<cs[j]; exp[si*256+t]=e; } }
    FILE*f=fopen(argv[2],"r"); int a,b,ns=0; int *sol=malloc(sizeof(int)*NV);
    while(fscanf(f,"%d %d",&a,&b)==2) sol[ns++]=a*256+b;
    fclose(f); fprintf(stderr,"solution size %d\n",ns);
    long ok=0; int first=-1;
    for(int v=0;v<NV;v++){
        int sv=v>>8, ev=exp[v], good=1;
        for(int q=0;q<ns;q++){
            int u=sol[q]; if(u==v){ good=0; break; }
            int cm=sup[sv]&sup[u>>8], t=pc(cm);
            if(t<=4) continue;
            int rq = t<=6 ? 1 : 2;
            if(pc((ev^exp[u])&cm) < rq){ good=0; break; }
        }
        if(good){ ok++; if(first<0){ first=v; } }
    }
    printf("candidates=%d  addable=%ld  first=%d\n",NV,ok,first);
    return 0;
}
