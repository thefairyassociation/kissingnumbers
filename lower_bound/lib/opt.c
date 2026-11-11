/* Spherical-code optimiser with threshold continuation.
 *
 *   ./opt n N steps seed [seedfile]        (env KISS_T sets the target, default 0.5)
 *
 * Minimises  E_t = sum_{i<j} max(0, g_ij - t)^2  by projected gradient descent
 * with an energy-adaptive step.  The threshold t is annealed: it starts at the
 * configuration's current maximum inner product and is lowered towards the
 * target each time E_t is driven to (numerical) zero, which is the continuation
 * scheme that works far better than attacking t = 1/2 directly.  Random kicks
 * restart from the best configuration when progress stalls. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int n, N;
static double t_target = 0.5;

static unsigned long long rs;
static double urand(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (rs>>11)*(1.0/9007199254740992.0); }
static double nrand(void){ double u=urand()+1e-18,v=urand(); return sqrt(-2*log(u))*cos(6.283185307179586*v); }

static void normalize(double *X){
    for(int i=0;i<N;i++){ double s=0,*x=X+(size_t)i*n;
        for(int k=0;k<n;k++) s+=x[k]*x[k];
        s=1.0/sqrt(s); for(int k=0;k<n;k++) x[k]*=s; }
}
static double engrad(const double *X, double *G, double t, double *mx){
    double E=0, m=-2;
    memset(G,0,sizeof(double)*(size_t)N*n);
    for(int i=0;i<N;i++){
        const double *xi=X+(size_t)i*n; double *gi=G+(size_t)i*n;
        for(int j=i+1;j<N;j++){
            const double *xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            if(g>m) m=g;
            double d=g-t;
            if(d>0){ E+=d*d; double c=2*d; double *gj=G+(size_t)j*n;
                for(int k=0;k<n;k++){ gi[k]+=c*xj[k]; gj[k]+=c*xi[k]; } }
        }
    }
    *mx=m; return E;
}
int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: opt n N steps seed [seedfile]\n"); return 1; }
    { const char*e=getenv("KISS_T"); if(e) t_target=atof(e); }
    n=atoi(argv[1]); N=atoi(argv[2]);
    long steps=atol(argv[3]); rs=strtoull(argv[4],0,10)*2862933555777941757ULL+3037000493ULL;
    for(int i=0;i<50;i++) urand();
    double *X=malloc(sizeof(double)*(size_t)N*n), *G=malloc(sizeof(double)*(size_t)N*n),
           *B=malloc(sizeof(double)*(size_t)N*n);
    for(size_t i=0;i<(size_t)N*n;i++) X[i]=nrand();
    if(argc>5){ FILE*f=fopen(argv[5],"r"); if(!f){perror("seed");return 1;}
        size_t cnt=0; while(cnt<(size_t)N*n && fscanf(f,"%lf",&X[cnt])==1) cnt++; fclose(f); }
    normalize(X);
    double mx, t=0;
    engrad(X,G,0.0,&mx);
    t = mx > t_target ? mx : t_target;          /* start feasible, then squeeze */
    double best=mx, lr=0.02, Eprev=1e300;
    memcpy(B,X,sizeof(double)*(size_t)N*n);
    long stall=0;
    for(long s=0;s<steps;s++){
        double E=engrad(X,G,t,&mx);
        if(mx<best-1e-13){ best=mx; memcpy(B,X,sizeof(double)*(size_t)N*n); stall=0;
            if(best<=t_target+1e-13){ fprintf(stderr,"FEASIBLE step %ld max=%.17g\n",s,best); break; } }
        else stall++;
        if(E<1e-22){                              /* satisfied: tighten the threshold */
            double gap=t-t_target;
            if(gap<=1e-13){ fprintf(stderr,"target met step %ld max=%.17g\n",s,mx); break; }
            t -= gap*0.15 + 1e-9; if(t<t_target) t=t_target;
            lr=0.02; Eprev=1e300; continue;
        }
        for(int i=0;i<N;i++){
            double *x=X+(size_t)i*n,*g=G+(size_t)i*n,d=0;
            for(int k=0;k<n;k++) d+=g[k]*x[k];
            for(int k=0;k<n;k++) x[k]-=lr*(g[k]-d*x[k]);
        }
        normalize(X);
        if(E<Eprev) lr*=1.03; else lr*=0.7;
        if(lr<1e-10) lr=1e-10;
        if(lr>0.5) lr=0.5;
        Eprev=E;
        if(stall>6000){                            /* kick from the best so far */
            memcpy(X,B,sizeof(double)*(size_t)N*n);
            double kick=0.01+0.06*urand();
            for(int i=0;i<N;i++){ double *x=X+(size_t)i*n;
                if(urand()<0.3) for(int k=0;k<n;k++) x[k]+=kick*nrand(); }
            normalize(X); double m2; engrad(X,G,0.0,&m2);
            t = m2>t_target ? m2 : t_target; lr=0.02; Eprev=1e300; stall=0;
        }
    }
    printf("n=%d N=%d best_max_inner=%.17g feasible=%d\n",n,N,best,best<=t_target+1e-12);
    char fn[512];
    if(argc>5) snprintf(fn,sizeof fn,"%s.n%d.N%d.s%s.out",argv[5],n,N,argv[4]);
    else snprintf(fn,sizeof fn,"%s.out",argv[4]);
    FILE*f=fopen(fn,"w");
    fprintf(f,"# n=%d N=%d max_inner=%.17g\n",n,N,best);
    for(int i=0;i<N;i++){ for(int k=0;k<n;k++) fprintf(f,"%.17g ",B[(size_t)i*n+k]); fputc('\n',f); }
    fclose(f);
    return 0;
}
