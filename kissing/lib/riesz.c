/* Riesz-energy continuation for spherical codes.
 *
 *   ./riesz n N steps seed [seedfile]
 *
 * Minimises  E_s = sum_{i<j} ||x_i - x_j||^{-s}  on the sphere for a geometrically
 * increasing sequence of exponents s.  As s -> infinity the minimisers approach
 * best-packing configurations.
 *
 * Inner solve at each s: backtracking gradient descent on the sphere
 * (retraction = row-normalise) by default -- this is what the committed riesz2
 * binary did, and what every legacy script in kissing/lib expects.  Set
 * KISS_SOLVER=lbfgs for limited-memory BFGS with a strong-Wolfe line search, or
 * KISS_SOLVER=adam for the published Takhanov-Assylbekov-Yun exponent schedule
 * with manifold Adam updates.
 *
 * Energy/gradient: one BLAS dgemm for the Gram matrix per evaluation (the old
 * code formed every inner product twice with O(N^2 n) loops), then an OpenMP
 * pair loop.  The Euclidean Riesz gradient is identical to the previous
 * implementation after the 1/E log-energy scaling, so seeds and checkpoints
 * keep the same meaning.
 *
 *   gcc -O3 -fopenmp -o riesz riesz.c -lopenblas -lm
 *
 * Env: KISS_JIT, KISS_S0, KISS_SMUL, KISS_SMAX, KISS_THREADS, KISS_SOLVER,
 *      KISS_M (L-BFGS memory), KISS_POLISH, KISS_ADAM_POLISH,
 *      KISS_ADAM_EPS, KISS_PENALTY_TARGET,
 *      KISS_SELFTEST, KISS_PROFILE, KISS_FAITHFUL, KISS_FAITHFUL_EXTRA.
 *
 * Boolean flags (KISS_FAITHFUL, KISS_POLISH, KISS_ADAM_POLISH,
 * KISS_ADAM_POLISH_ONLY, KISS_ADAM_RAW, KISS_PENALTY_ONLY, KISS_PROFILE,
 * KISS_SELFTEST) are all "present and non-zero": FLAG=0 disables.
 *
 * KISS_FAITHFUL=1 is an explicit source-fidelity mode for the published
 * 841-point search.  It is intentionally opt-in: legacy seeded runs retain
 * their historical Gaussian-jitter, schedule-scaling, and penalty-polish
 * behaviour.  In faithful mode the input seed supplies the exact 840-point
 * core, the final row is replaced by one uniform random hypercube extra, raw
 * Adam is forced, and the 35,000-step published schedule is used verbatim.
 * Set KISS_ADAM_POLISH_ONLY=1 to run the separate authors' polish schedule;
 * polish_lr[] already stores their *effective* rates, i.e. their schedule after
 * polish_841.py's own group["lr"]=lr/10, so no further division is applied.
 * Faithful mode never falls through to penalty polishing.
 *
 * The authors' %.10f Gram -> rank-12 eigendecomposition handoff sits between
 * their search and their polish, and is not implemented in C.  Faithful mode
 * therefore refuses search-and-polish in one process; run
 * prepare_841_polish.py in between.  gram_handoff=0 in a candidate header
 * means only that *this process* did not perform it -- for a polish-only run
 * the caller is responsible for having prepared the input.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int n, N;
static unsigned long long rs;
static double LOGSCALE;
static int do_profile;
static double t_gemm, t_pairs, t_cx;
static long n_engrad;

static double *Gram, *Cmat, *GX, *rowsum;
static double adam_eps=1e-8;
static int faithful_mode;
static double pair_r2_floor=1e-12;
static int faithful_failure;

/* Boolean environment flags: present *and* non-zero.  Using getenv()!=NULL
 * alone made KISS_ADAM_POLISH=0 (and KISS_SELFTEST=0) enable the feature. */
static int env_flag(const char *name){
    const char *e=getenv(name);
    return e!=NULL && atoi(e)!=0;
}

static int finite_block(const double *x, size_t count){
    for(size_t i=0;i<count;i++) if(!isfinite(x[i])) return 0;
    return 1;
}

static int faithful_guard(const char *phase, const char *name,
                          const double *x, size_t count){
    if(!faithful_mode || finite_block(x,count)) return 1;
    faithful_failure=1;
    fprintf(stderr,"faithful nonfinite guard: %s (%s)\n",phase,name);
    return 0;
}

static int faithful_rows_guard(const char *phase, const char *name,
                               const double *x){
    if(!faithful_mode) return 1;
    for(int i=0;i<N;i++){
        double q=0.0;
        for(int k=0;k<n;k++) q+=x[(size_t)i*n+k]*x[(size_t)i*n+k];
        if(!isfinite(q) || !(q>1e-30)){
            faithful_failure=1;
            fprintf(stderr,"faithful row guard: %s (%s row %d)\n",phase,name,i);
            return 0;
        }
    }
    return 1;
}

static int faithful_breakdown(const char *phase){
    if(faithful_mode){
        faithful_failure=1;
        fprintf(stderr,"faithful numerical breakdown: %s\n",phase);
        return -2;
    }
    return -1;
}

static unsigned long long uraw(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
static double urand(void){ return (uraw()>>11)*(1.0/9007199254740992.0); }
static double nrand(void){ double u=urand()+1e-18,v=urand(); return sqrt(-2*log(u))*cos(6.283185307179586*v); }

static double wall(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

static void normalize(double *X){
    for(int i=0;i<N;i++){ double s=0,*x=X+(size_t)i*n;
        for(int k=0;k<n;k++) s+=x[k]*x[k];
        if(!(s>0)) s=1;
        s=1.0/sqrt(s); for(int k=0;k<n;k++) x[k]*=s; }
}

static void project_tangent(const double *X, double *G){
    for(int i=0;i<N;i++){
        const double *x=X+(size_t)i*n; double *g=G+(size_t)i*n, d=0;
        for(int k=0;k<n;k++) d+=g[k]*x[k];
        for(int k=0;k<n;k++) g[k]-=d*x[k];
    }
}

static void retract(double *Y, const double *X, const double *P, double a){
    size_t Nn=(size_t)N*n;
    for(size_t t=0;t<Nn;t++) Y[t]=X[t]+a*P[t];
    normalize(Y);
}

static void gram_dgemm(const double *X){
    /* Gram = X X^T, X is N x n row-major. */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, n, 1.0, X, n, X, n, 0.0, Gram, N);
}

static int loss_ip; /* KISS_LOSS=ip: minimise smooth-max inner product */

static double engrad_riesz(const double *X, double *G, double s, double *mx);
static double engrad_ip(const double *X, double *G, double s, double *mx);

static double engrad(const double *X, double *G, double s, double *mx){
    return loss_ip ? engrad_ip(X,G,s,mx) : engrad_riesz(X,G,s,mx);
}

/* Log-energy LOGSCALE = log sum_{i<j} r_ij^{-s} and, if G!=NULL, its Euclidean
 * gradient (already divided so it matches d log E / dX of the original code).
 * One Gram evaluation.  Returns LOGSCALE, or -1 on NaN. */
static double engrad_riesz(const double *X, double *G, double s, double *mx){
    double t0=0, t1=0;
    if(do_profile) t0=wall();
    gram_dgemm(X);
    if(do_profile){ t1=wall(); t_gemm+=t1-t0; t0=t1; }

    double m=-2, r2min=1e300;
#ifdef _OPENMP
#pragma omp parallel
    {
        double lm=-2, lr=1e300;
#pragma omp for nowait schedule(static)
        for(int i=0;i<N;i++){
            const double *gi=Gram+(size_t)i*N;
            for(int j=i+1;j<N;j++){
                double g=gi[j];
                if(g>lm) lm=g;
                double r2=2-2*g; if(r2<pair_r2_floor) r2=pair_r2_floor;
                if(r2<lr) lr=r2;
            }
        }
#pragma omp critical
        { if(lm>m) m=lm; if(lr<r2min) r2min=lr; }
    }
#else
    for(int i=0;i<N;i++){
        const double *gi=Gram+(size_t)i*N;
        for(int j=i+1;j<N;j++){
            double g=gi[j];
            if(g>m) m=g;
            double r2=2-2*g; if(r2<pair_r2_floor) r2=pair_r2_floor;
            if(r2<r2min) r2min=r2;
        }
    }
#endif
    *mx=m;
    if(!(m>-1.5)) return -1.0;
    n_engrad++;

    int want_grad = G!=NULL;
    if(want_grad) memset(Cmat, 0, sizeof(double)*(size_t)N*N);

    double E=0, halfs=0.5*s;
    /* Legacy mode drops pair contributions below exp(-40); faithful mode keeps
     * every term, matching torch.logsumexp's source semantics. */
    const double log_cut=40.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:E) schedule(static)
#endif
    for(int i=0;i<N;i++){
        const double *gi=Gram+(size_t)i*N;
        double *ci=want_grad ? Cmat+(size_t)i*N : NULL;
        for(int j=i+1;j<N;j++){
            double g=gi[j];
            double r2=2-2*g; if(r2<pair_r2_floor) r2=pair_r2_floor;
            double logr=log(r2/r2min);
            if(!faithful_mode && halfs*logr>log_cut) continue;
            double e=exp(-halfs*logr);
            E+=e;
            if(want_grad){
                double c=s*e/r2;
                ci[j]=c;
                Cmat[(size_t)j*N+i]=c;
            }
        }
    }
    if(do_profile){ t1=wall(); t_pairs+=t1-t0; t0=t1; }

    if(!(E>0)) return -1.0;
    LOGSCALE = log(E) - halfs*log(r2min);

    if(want_grad){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int i=0;i<N;i++){
            const double *ci=Cmat+(size_t)i*N; double rs_=0;
            for(int j=0;j<N;j++) rs_+=ci[j];
            rowsum[i]=rs_;
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, n, N, 1.0, Cmat, N, X, n, 0.0, GX, n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int i=0;i<N;i++){
            double rs_=rowsum[i];
            const double *gx=GX+(size_t)i*n, *x=X+(size_t)i*n;
            double *g=G+(size_t)i*n;
            for(int k=0;k<n;k++) g[k]=gx[k]-rs_*x[k];
        }
        cblas_dscal((size_t)N*n, 1.0/E, G, 1);
        if(do_profile){ t1=wall(); t_cx+=t1-t0; }
    }
    return LOGSCALE;
}

/* Smooth max of off-diagonal inner products: L = (1/s) log sum_{i<j} exp(s g_ij).
 * As s -> infinity this is exactly the max inner product.  One dgemm. */
static double engrad_ip(const double *X, double *G, double s, double *mx){
    double t0=0, t1=0;
    if(do_profile) t0=wall();
    gram_dgemm(X);
    if(do_profile){ t1=wall(); t_gemm+=t1-t0; t0=t1; }

    double m=-2;
#ifdef _OPENMP
#pragma omp parallel
    {
        double lm=-2;
#pragma omp for nowait schedule(static)
        for(int i=0;i<N;i++){
            const double *gi=Gram+(size_t)i*N;
            for(int j=i+1;j<N;j++) if(gi[j]>lm) lm=gi[j];
        }
#pragma omp critical
        { if(lm>m) m=lm; }
    }
#else
    for(int i=0;i<N;i++){
        const double *gi=Gram+(size_t)i*N;
        for(int j=i+1;j<N;j++) if(gi[j]>m) m=gi[j];
    }
#endif
    *mx=m;
    if(!(m>-1.5)) return -1.0;
    n_engrad++;

    int want_grad=G!=NULL;
    if(want_grad) memset(Cmat,0,sizeof(double)*(size_t)N*N);
    double Z=0, ss=(s>1e-12?s:1e-12);
#ifdef _OPENMP
#pragma omp parallel for reduction(+:Z) schedule(static)
#endif
    for(int i=0;i<N;i++){
        const double *gi=Gram+(size_t)i*N;
        double *ci=want_grad ? Cmat+(size_t)i*N : NULL;
        for(int j=i+1;j<N;j++){
            double a=ss*(gi[j]-m);
            if(a<-40) continue;
            double e=exp(a);
            Z+=e;
            if(want_grad){ ci[j]=e; Cmat[(size_t)j*N+i]=e; }
        }
    }
    if(do_profile){ t1=wall(); t_pairs+=t1-t0; t0=t1; }
    if(!(Z>0)) return -1.0;
    LOGSCALE = m + log(Z)/ss; /* ≈ max inner product */
    if(want_grad){
        cblas_dscal((size_t)N*N, 1.0/Z, Cmat, 1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, n, N, 1.0, Cmat, N, X, n, 0.0, G, n);
        if(do_profile){ t1=wall(); t_cx+=t1-t0; }
    }
    return LOGSCALE;
}

/* Reference O(N^2 n) engrad from the previous code, for KISS_SELFTEST. */
static double engrad_ref(const double *X, double *G, double s, double *mx){
    double E=0,m=-2,r2min=1e300;
    for(int i=0;i<N;i++){
        const double *xi=X+(size_t)i*n;
        for(int j=i+1;j<N;j++){
            const double *xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            if(g>m) m=g;
            double r2=2-2*g; if(r2<1e-12) r2=1e-12;
            if(r2<r2min) r2min=r2;
        }
    }
    *mx=m;
    if(!(m>-1.5)) return -1.0;
    memset(G,0,sizeof(double)*(size_t)N*n);
    for(int i=0;i<N;i++){
        const double *xi=X+(size_t)i*n; double *gi=G+(size_t)i*n;
        for(int j=i+1;j<N;j++){
            const double *xj=X+(size_t)j*n; double g=0;
            for(int k=0;k<n;k++) g+=xi[k]*xj[k];
            double r2=2-2*g; if(r2<1e-12) r2=1e-12;
            double e=pow(r2/r2min,-s/2); E+=e;
            double c=s*e/r2;
            double *gj=G+(size_t)j*n;
            for(int k=0;k<n;k++){ double dk=xi[k]-xj[k]; gi[k]-=c*dk; gj[k]+=c*dk; }
        }
    }
    if(E>0) for(size_t t=0;t<(size_t)N*n;t++) G[t]/=E;
    LOGSCALE = log(E) - (s / 2) * log(r2min);
    return E;
}

static int selftest(void){
    n=6; N=17;
    size_t Nn=(size_t)N*n, NN=(size_t)N*N;
    Gram=malloc(sizeof(double)*NN); Cmat=malloc(sizeof(double)*NN);
    GX=malloc(sizeof(double)*Nn); rowsum=malloc(sizeof(double)*N);
    double *X=malloc(sizeof(double)*Nn), *G0=malloc(sizeof(double)*Nn), *G1=malloc(sizeof(double)*Nn);
    rs=12345;
    for(size_t t=0;t<Nn;t++) X[t]=nrand();
    normalize(X);
    double mx0,mx1,s=4.7;
    double Eref=engrad_ref(X,G0,s,&mx0);
    double lsref=LOGSCALE;
    double ls=engrad(X,G1,s,&mx1);
    double eg=0, gg=0, gn=0;
    for(size_t t=0;t<Nn;t++){
        double d=G0[t]-G1[t]; gg+=d*d; gn+=G0[t]*G0[t];
    }
    eg=gg;
    int ok = fabs(mx0-mx1)<1e-12 && fabs(ls-lsref)<1e-9 && sqrt(eg)<1e-8* (1+sqrt(gn));
    fprintf(stderr,"selftest mx ref=%.17g blas=%.17g  logE ref=%.17g blas=%.17g  |G|=%.3g |dG|=%.3g  %s\n",
            mx0,mx1,lsref,ls,sqrt(gn),sqrt(eg), ok?"OK":"FAIL");
    (void)Eref;
    free(X); free(G0); free(G1);
    return ok?0:1;
}

/* Penalty  sum_{i<j} max(0, g_ij - t)^2  and Euclidean gradient, one Gram. */
static double engrad_pen(const double *X, double *G, double t, double *mx){
    gram_dgemm(X);
    double m=-2, E=0;
#ifdef _OPENMP
#pragma omp parallel
    {
        double lm=-2, lE=0;
#pragma omp for nowait schedule(static)
        for(int i=0;i<N;i++){
            const double *gi=Gram+(size_t)i*N;
            for(int j=i+1;j<N;j++){
                double g=gi[j]; if(g>lm) lm=g;
                double d=g-t; if(d>0) lE+=d*d;
            }
        }
#pragma omp critical
        { if(lm>m) m=lm; E+=lE; }
    }
#else
    for(int i=0;i<N;i++){
        const double *gi=Gram+(size_t)i*N;
        for(int j=i+1;j<N;j++){
            double g=gi[j]; if(g>m) m=g;
            double d=g-t; if(d>0) E+=d*d;
        }
    }
#endif
    *mx=m;
    if(!G) return E;
    memset(Cmat,0,sizeof(double)*(size_t)N*N);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        const double *gi=Gram+(size_t)i*N;
        double *ci=Cmat+(size_t)i*N;
        for(int j=i+1;j<N;j++){
            double d=gi[j]-t;
            if(d>0){ double c=2*d; ci[j]=c; Cmat[(size_t)j*N+i]=c; }
        }
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, n, N, 1.0, Cmat, N, X, n, 0.0, G, n);
    return E;
}

/* ---------- L-BFGS (Nocedal two-loop) + strong Wolfe on the sphere ---------- */
#define MEMMAX 32
static int MEM=17, nmem, hpos;
static double *Smem, *Ymem, rho[MEMMAX];

static int lidx(int k){ /* k=0 newest */
    int i=hpos-1-k; i%=MEM; if(i<0) i+=MEM; return i;
}

static void lbfgs_reset(void){ nmem=0; hpos=0; }

static void two_loop(const double *g, double *p, int Nn){
    cblas_dcopy(Nn, g, 1, p, 1);
    if(nmem<=0){ cblas_dscal(Nn, -1.0, p, 1); return; }
    double alpha[MEMMAX];
    for(int k=0;k<nmem;k++){
        int i=lidx(k);
        alpha[k]=rho[i]*cblas_ddot(Nn, Smem+(size_t)i*Nn, 1, p, 1);
        cblas_daxpy(Nn, -alpha[k], Ymem+(size_t)i*Nn, 1, p, 1);
    }
    {   int i=lidx(0);
        double ys=1.0/rho[i];
        double yy=cblas_ddot(Nn, Ymem+(size_t)i*Nn, 1, Ymem+(size_t)i*Nn, 1);
        double gamma=(yy>0 && ys>0) ? ys/yy : 1.0;
        cblas_dscal(Nn, gamma, p, 1);
    }
    for(int k=nmem-1;k>=0;k--){
        int i=lidx(k);
        double beta=rho[i]*cblas_ddot(Nn, Ymem+(size_t)i*Nn, 1, p, 1);
        cblas_daxpy(Nn, alpha[k]-beta, Smem+(size_t)i*Nn, 1, p, 1);
    }
    cblas_dscal(Nn, -1.0, p, 1);
}

static double max_row_norm(const double *P){
    double m=0;
    for(int i=0;i<N;i++){
        const double *p=P+(size_t)i*n; double s=0;
        for(int k=0;k<n;k++) s+=p[k]*p[k];
        s=sqrt(s); if(s>m) m=s;
    }
    return m;
}

/* Strong Wolfe on f(retract(X + a P)).  Returns 1 on success. */
static int wolfe(const double *X, const double *G, const double *P,
                 double f0, double dphi0, double s,
                 double *Y, double *GY, double *fnew, double *mx,
                 double *alpha_io, int maxit){
    const double c1=1e-4, c2=0.9;
    double a= *alpha_io, alo=0, ahi=0, flo=f0;
    int have_hi=0, Nn=(int)((size_t)N*n);
    if(!(dphi0<0)) return 0;
    for(int it=0; it<maxit; it++){
        retract(Y,X,P,a);
        double f=engrad(Y,GY,s,mx);
        if(f<0) return 0;
        project_tangent(Y,GY);
        double dphi=cblas_ddot(Nn, GY, 1, P, 1);
        int armijo = (f <= f0 + c1*a*dphi0);
        if(!armijo || (it>0 && f>=flo)){
            ahi=a; have_hi=1;
            goto zoom;
        }
        if(fabs(dphi) <= -c2*dphi0){ *alpha_io=a; *fnew=f; return 1; }
        if(dphi>=0){ ahi=alo; alo=a; flo=f; have_hi=1; goto zoom; }
        alo=a; flo=f;
        a*=2.0; if(a>50) a=50;
    }
    return 0;
zoom:
    for(int it=0; it<maxit; it++){
        if(!have_hi) return 0;
        a=0.5*(alo+ahi);
        if(!(fabs(ahi-alo)>1e-16*(1+fabs(alo)))) break;
        retract(Y,X,P,a);
        double f=engrad(Y,GY,s,mx);
        if(f<0) return 0;
        project_tangent(Y,GY);
        double dphi=cblas_ddot(Nn, GY, 1, P, 1);
        if(!(f <= f0 + c1*a*dphi0) || f>=flo){ ahi=a; }
        else{
            if(fabs(dphi) <= -c2*dphi0){ *alpha_io=a; *fnew=f; return 1; }
            if(dphi*(ahi-alo)>=0) ahi=alo;
            alo=a; flo=f;
        }
    }
    /* Armijo fallback */
    a= *alpha_io;
    for(int b=0;b<20;b++){
        retract(Y,X,P,a);
        double f=engrad(Y,GY,s,mx);
        if(f<0) return 0;
        if(f < f0){ project_tangent(Y,GY); *alpha_io=a; *fnew=f; return 1; }
        a*=0.5;
    }
    return 0;
}

static int lbfgs_stage(double *X, double *G, double *B, double *Y, double *GY, double *P,
                       double s, long maxit, double *best){
    int Nn=(int)((size_t)N*n);
    lbfgs_reset();
    double mx, f=engrad(X,G,s,&mx);
    if(f<0) return -1;
    project_tangent(X,G);
    if(mx<*best-1e-13){ *best=mx; memcpy(B,X,sizeof(double)*(size_t)Nn);
        if(*best<=0.5+1e-13) return 1; }
    double alpha=0.05, f_prev=f; int stall=0;
    for(long it=0; it<maxit; it++){
        two_loop(G,P,Nn);
        project_tangent(X,P);
        double dphi=cblas_ddot(Nn, G, 1, P, 1);
        if(!(dphi<0)){
            cblas_dcopy(Nn,G,1,P,1); cblas_dscal(Nn,-1.0,P,1);
            dphi=cblas_ddot(Nn, G, 1, P, 1);
        }
        double mrn=max_row_norm(P);
        if(nmem==0) alpha=0.05/(sqrt(cblas_ddot(Nn,P,1,P,1)/(Nn>0?Nn:1))+1e-16);
        else if(alpha<=0 || alpha>4) alpha=1.0;
        if(mrn>0 && alpha*mrn>0.5) alpha=0.5/mrn;

        /* wolfe writes Y, GY; X and G stay the previous iterate (s = Y-X, y = GY-G). */
        double fnew=f, mxn=mx;
        if(!wolfe(X,G,P,f,dphi,s,Y,GY,&fnew,&mxn,&alpha,20)){
            if(it==0){
                alpha=0.05;
                int acc=0;
                for(int b=0;b<16;b++){
                    retract(Y,X,G,-alpha);
                    double ft=engrad(Y,GY,s,&mxn);
                    if(ft>=0 && ft<f){ memcpy(X,Y,sizeof(double)*(size_t)Nn); memcpy(G,GY,sizeof(double)*(size_t)Nn);
                        project_tangent(X,G); f=ft; acc=1; break; }
                    alpha*=0.5;
                }
                if(!acc) break;
                if(mxn<*best-1e-13){ *best=mxn; memcpy(B,X,sizeof(double)*(size_t)Nn);
                    if(*best<=0.5+1e-13) return 1; }
                continue;
            }
            break; /* stage converged */
        }
        int slot=hpos%MEM;
        double *ss=Smem+(size_t)slot*Nn, *yy=Ymem+(size_t)slot*Nn;
        for(int t=0;t<Nn;t++){ ss[t]=Y[t]-X[t]; yy[t]=GY[t]-G[t]; }
        double ys=cblas_ddot(Nn,yy,1,ss,1);
        double yy2=cblas_ddot(Nn,yy,1,yy,1);
        if(ys>1e-10*sqrt(yy2*cblas_ddot(Nn,ss,1,ss,1)+1e-300) && ys>0){
            rho[slot]=1.0/ys;
            if(nmem<MEM) nmem++;
            hpos++;
        }
        memcpy(X,Y,sizeof(double)*(size_t)Nn);
        memcpy(G,GY,sizeof(double)*(size_t)Nn);
        f=fnew;
        if(mxn<*best-1e-13){ *best=mxn; memcpy(B,X,sizeof(double)*(size_t)Nn);
            if(*best<=0.5+1e-13) return 1; }
        double gnorm=sqrt(cblas_ddot(Nn,G,1,G,1)/(Nn>0?Nn:1));
        if(gnorm<1e-10) break;
        if(fabs(f-f_prev)<1e-12*(1.0+fabs(f))) stall++; else stall=0;
        f_prev=f;
        if(stall>=8) break;
        alpha=1.0;
    }
    return 0;
}

/* Old inner solver: projected GD with Armijo backtracking, now on the BLAS engrad. */
static int gd_stage(double *X, double *G, double *B, double *Y, double s, long per, double *best){
    double lr=0.05;
    size_t Nn=(size_t)N*n;
    for(long it=0; it<per; it++){
        double mx,f=engrad(X,G,s,&mx);
        if(f<0) return -1;
        if(mx<*best-1e-13){ *best=mx; memcpy(B,X,sizeof(double)*Nn);
            if(*best<=0.5+1e-13) return 1; }
        project_tangent(X,G);
        int accepted=0;
        for(int back=0; back<24; back++){
            retract(Y,X,G,-lr);
            double mx2, f2=engrad(Y,NULL,s,&mx2);
            if(f2<0) break;
            if(f2<f){ memcpy(X,Y,sizeof(double)*Nn); accepted=1; break; }
            lr*=0.5;
        }
        if(!accepted){ lr=0.05; break; }
        lr*=1.6; if(lr>0.5) lr=0.5;
    }
    return 0;
}

/* Adam continuation using the published 841-point exponent/LR schedule.
 * Moments persist across stages, as in torch.optim.Adam. */
static int adam_stage(double *X, double *G, double *B, double *M1, double *M2,
                      double s, long maxit, double lr, long *adam_it,
                      double *beta1_pow, double *beta2_pow, double *best){
    const double beta1=0.9, beta2=0.999;
    size_t Nn=(size_t)N*n;
    if(!faithful_guard("polish stage entry","X",X,Nn) ||
       !faithful_rows_guard("polish stage entry","X",X) ||
       !faithful_guard("polish stage entry","M1",M1,Nn) ||
       !faithful_guard("polish stage entry","M2",M2,Nn)) return -2;
    for(long it=0; it<maxit; it++){
        if(!faithful_guard("polish step input","X",X,Nn) ||
           !faithful_guard("polish step input","M1",M1,Nn) ||
           !faithful_guard("polish step input","M2",M2,Nn)) return -2;
        double mx,f=engrad(X,G,s,&mx);
        if(f<0 || !isfinite(f) || !isfinite(mx)) return faithful_breakdown("polish loss");
        if(!faithful_guard("polish gradient","G",G,Nn)) return -2;
        if(!faithful_mode && mx<*best-1e-13){
            *best=mx; memcpy(B,X,sizeof(double)*Nn);
            if(*best<=0.5+1e-13) return 1;
        }
        project_tangent(X,G);
        if(!faithful_guard("polish tangent gradient","G",G,Nn)) return -2;
        *beta1_pow *= beta1;
        *beta2_pow *= beta2;
        double c1=1.0/(1.0-*beta1_pow), c2=1.0/(1.0-*beta2_pow);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(size_t t=0;t<Nn;t++){
            double g=G[t];
            M1[t]=beta1*M1[t]+(1.0-beta1)*g;
            M2[t]=beta2*M2[t]+(1.0-beta2)*g*g;
            X[t]-=lr*(M1[t]*c1)/(sqrt(M2[t]*c2)+adam_eps);
        }
        normalize(X);
        if(!faithful_guard("polish step update","X",X,Nn) ||
           !faithful_rows_guard("polish step update","X",X) ||
           !faithful_guard("polish step update","M1",M1,Nn) ||
           !faithful_guard("polish step update","M2",M2,Nn)) return -2;
        (*adam_it)++;
    }
    double mx,f=engrad(X,NULL,s,&mx);
    if(f<0 || !isfinite(f) || !isfinite(mx)) return faithful_breakdown("polish stage final loss");
    if(!faithful_guard("polish stage final","X",X,Nn) ||
       !faithful_rows_guard("polish stage final","X",X)) return -2;
    if(faithful_mode){
        /* Same rule as adam_raw_stage: the authors' polish returns the final
         * state of the stage, not the best transient.  Without this the
         * candidate buffer B kept whatever came in whenever the polish did not
         * improve max-IP, so a faithful polish serialized its own *input* and
         * silently discarded every update it had just computed. */
        *best=mx; memcpy(B,X,sizeof(double)*Nn);
        if(!faithful_guard("polish stage result","B",B,Nn) ||
           !faithful_rows_guard("polish stage result","B",B)) return -2;
        return 0;
    }
    if(mx<*best-1e-13){
        *best=mx; memcpy(B,X,sizeof(double)*Nn);
        if(!faithful_guard("polish stage result","B",B,Nn) ||
           !faithful_rows_guard("polish stage result","B",B)) return -2;
        if(*best<=0.5+1e-13) return 1;
    }
    return 0;
}

/* Faithful port of search_841_riesz.py's search phase.  Its Adam parameters
 * are unconstrained: the loss normalises a view Z=X/||X||, but opt.step()
 * updates raw X and does not retract it.  The chain-rule gradient is
 * (I-ZZ^T) grad_Z / ||X||.  This differs materially from manifold Adam because
 * Adam's coordinatewise preconditioner is not rotation/radial invariant. */
static int adam_raw_stage(double *X, double *G, double *Z, double *B,
                          double *M1, double *M2, double *norms,
                          double s, long maxit, double lr, long *adam_it,
                          double *beta1_pow, double *beta2_pow, double *best){
    const double beta1=0.9, beta2=0.999;
    size_t Nn=(size_t)N*n;
    if(!faithful_guard("raw stage entry","X",X,Nn) ||
       !faithful_rows_guard("raw stage entry","X",X) ||
       !faithful_guard("raw stage entry","M1",M1,Nn) ||
       !faithful_guard("raw stage entry","M2",M2,Nn)) return -2;
    for(long it=0;it<maxit;it++){
        if(!faithful_guard("raw step input","X",X,Nn) ||
           !faithful_guard("raw step input","M1",M1,Nn) ||
           !faithful_guard("raw step input","M2",M2,Nn)) return -2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int i=0;i<N;i++){
            const double *x=X+(size_t)i*n; double *z=Z+(size_t)i*n;
            double q=0; for(int k=0;k<n;k++) q+=x[k]*x[k];
            double r=sqrt(q); if(!(r>1e-12)) r=1e-12;
            norms[i]=r; for(int k=0;k<n;k++) z[k]=x[k]/r;
        }
        if(!faithful_guard("raw normalized view","Z",Z,Nn) ||
           !faithful_guard("raw normalized view","norms",norms,(size_t)N)) return -2;
        double mx,f=engrad(Z,G,s,&mx);
        if(f<0 || !isfinite(f) || !isfinite(mx)) return faithful_breakdown("raw loss");
        if(!faithful_guard("raw gradient","G",G,Nn)) return -2;
        if(mx<*best-1e-13){
            *best=mx;
            if(!faithful_mode) memcpy(B,Z,sizeof(double)*Nn);
            if(!faithful_mode && *best<=0.5+1e-13) return 1;
        }
        project_tangent(Z,G);
        if(!faithful_guard("raw tangent gradient","G",G,Nn)) return -2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int i=0;i<N;i++){
            double invr=1.0/norms[i], *g=G+(size_t)i*n;
            for(int k=0;k<n;k++) g[k]*=invr;
        }
        *beta1_pow *= beta1;
        *beta2_pow *= beta2;
        double c1=1.0/(1.0-*beta1_pow), c2=1.0/(1.0-*beta2_pow);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(size_t t=0;t<Nn;t++){
            double g=G[t];
            M1[t]=beta1*M1[t]+(1.0-beta1)*g;
            M2[t]=beta2*M2[t]+(1.0-beta2)*g*g;
            X[t]-=lr*(M1[t]*c1)/(sqrt(M2[t]*c2)+adam_eps);
        }
        if(!faithful_guard("raw step update","X",X,Nn) ||
           !faithful_guard("raw step update","M1",M1,Nn) ||
           !faithful_guard("raw step update","M2",M2,Nn)) return -2;
        (*adam_it)++;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        const double *x=X+(size_t)i*n; double *z=Z+(size_t)i*n;
        double q=0; for(int k=0;k<n;k++) q+=x[k]*x[k];
        double r=sqrt(q); if(!(r>1e-12)) r=1e-12;
        for(int k=0;k<n;k++) z[k]=x[k]/r;
    }
    if(!faithful_guard("raw stage final view","Z",Z,Nn) ||
       !faithful_rows_guard("raw stage final view","Z",Z)) return -2;
    double mx,f=engrad(Z,NULL,s,&mx);
    if(f<0 || !isfinite(f) || !isfinite(mx)) return faithful_breakdown("raw stage final loss");
    if(faithful_mode){
        /* Authors' search accounts for the final state of each macro, not
         * the best transient observed during it. */
        *best=mx; memcpy(B,Z,sizeof(double)*Nn);
        if(!faithful_guard("raw stage result","B",B,Nn) ||
           !faithful_rows_guard("raw stage result","B",B)) return -2;
        return 0;
    }
    if(mx<*best-1e-13){
        *best=mx; memcpy(B,Z,sizeof(double)*Nn);
        if(*best<=0.5+1e-13) return 1;
    }
    return 0;
}

/* Penalty L-BFGS (same two-loop, Armijo on E_t). */
static int lbfgs_pen_stage(double *X, double *G, double *B, double *Y, double *GY, double *P,
                           double t, long maxit, double *best){
    int Nn=(int)((size_t)N*n);
    lbfgs_reset();
    double mx, f=engrad_pen(X,G,t,&mx);
    project_tangent(X,G);
    if(mx<*best-1e-13){ *best=mx; memcpy(B,X,sizeof(double)*(size_t)Nn);
        if(*best<=0.5+1e-13) return 1; }
    double alpha=0.05;
    for(long it=0; it<maxit; it++){
        two_loop(G,P,Nn);
        project_tangent(X,P);
        double dphi=cblas_ddot(Nn,G,1,P,1);
        if(!(dphi<0)){ cblas_dcopy(Nn,G,1,P,1); cblas_dscal(Nn,-1.0,P,1); dphi=cblas_ddot(Nn,G,1,P,1); }
        double mrn=max_row_norm(P);
        if(mrn>0 && alpha*mrn>0.4) alpha=0.4/mrn;
        int acc=0;
        double a=alpha;
        for(int b=0;b<24;b++){
            retract(Y,X,P,a);
            double mxn,f2=engrad_pen(Y,GY,t,&mxn);
            project_tangent(Y,GY);
            if(f2 <= f + 1e-4*a*dphi || f2<f){
                int slot=hpos%MEM;
                double *ss=Smem+(size_t)slot*Nn, *yy=Ymem+(size_t)slot*Nn;
                for(int u=0;u<Nn;u++){ ss[u]=Y[u]-X[u]; yy[u]=GY[u]-G[u]; }
                double ys=cblas_ddot(Nn,yy,1,ss,1), yy2=cblas_ddot(Nn,yy,1,yy,1);
                if(ys>1e-12 && yy2>0){ rho[slot]=1.0/ys; if(nmem<MEM) nmem++; hpos++; }
                memcpy(X,Y,sizeof(double)*(size_t)Nn); memcpy(G,GY,sizeof(double)*(size_t)Nn);
                f=f2; acc=1; alpha=a*1.2; if(alpha>1) alpha=1;
                if(mxn<*best-1e-13){ *best=mxn; memcpy(B,X,sizeof(double)*(size_t)Nn);
                    if(*best<=0.5+1e-13) return 1; }
                break;
            }
            a*=0.5;
        }
        if(!acc) break;
        if(f<1e-22) break;
        double gnorm=sqrt(cblas_ddot(Nn,G,1,G,1)/(Nn>0?Nn:1));
        if(gnorm<1e-12) break;
    }
    return 0;
}

int main(int argc,char**argv){
    if(env_flag("KISS_SELFTEST")) return selftest();
    if(argc<5){ fprintf(stderr,"usage: riesz n N steps seed [seedfile]\n"); return 1; }
    n=atoi(argv[1]); N=atoi(argv[2]); long steps=atol(argv[3]);
    rs=strtoull(argv[4],0,10)*2862933555777941757ULL+3037000493ULL;
    for(int i=0;i<40;i++) urand();

    int nthreads=0;
    { const char*e=getenv("KISS_THREADS"); if(e) nthreads=atoi(e); }
#ifdef _OPENMP
    if(nthreads<=0) nthreads=omp_get_max_threads();
    if(nthreads<=0) nthreads=4;
    omp_set_num_threads(nthreads);
#else
    nthreads=1;
#endif
    openblas_set_num_threads(1);

    /* Default is the geometric-homotopy GD that the committed riesz2 binary
     * implemented; every legacy script (rieszloop.sh, calib2.sh, calibrate.sh,
     * relaunch.sh, restart_all.sh) relies on it and none of them set
     * KISS_SOLVER.  L-BFGS and Adam are opt-in. */
    enum { SOLVER_LBFGS, SOLVER_GD, SOLVER_ADAM } solver=SOLVER_GD;
    { const char*e=getenv("KISS_SOLVER");
      if(e && !strcmp(e,"gd")) solver=SOLVER_GD;
      else if(e && !strcmp(e,"lbfgs")) solver=SOLVER_LBFGS;
      else if(e && !strcmp(e,"adam")) solver=SOLVER_ADAM;
      else if(e && *e){
          fprintf(stderr,"unknown KISS_SOLVER=%s (want gd, lbfgs or adam)\n",e);
          return 1;
      } }
    faithful_mode = env_flag("KISS_FAITHFUL");
    /* Faithful mode is an Adam protocol; it may still be requested without an
     * explicit KISS_SOLVER now that the unmarked default is GD. */
    if(faithful_mode && getenv("KISS_SOLVER")==NULL) solver=SOLVER_ADAM;
    int adam_polish_only=env_flag("KISS_ADAM_POLISH_ONLY");
    if(faithful_mode){
        if(n!=12 || N!=841){
            fprintf(stderr,"KISS_FAITHFUL currently requires n=12 and N=841\n");
            return 1;
        }
        if(solver!=SOLVER_ADAM){
            fprintf(stderr,"KISS_FAITHFUL requires KISS_SOLVER=adam (or no solver override)\n");
            return 1;
        }
        if(env_flag("KISS_PENALTY_ONLY")){
            fprintf(stderr,"KISS_FAITHFUL is incompatible with KISS_PENALTY_ONLY\n");
            return 1;
        }
        /* Search-then-polish in one process would polish the search's raw
         * coordinates, skipping the authors' %.10f Gram -> rank-12
         * eigendecomposition handoff.  That is a third protocol, neither their
         * search nor their polish, so faithful mode refuses it rather than
         * running it quietly.  The handoff is lossy and part of the source. */
        if(env_flag("KISS_ADAM_POLISH") && !adam_polish_only){
            fprintf(stderr,
                "KISS_FAITHFUL cannot run search and polish in one process: that\n"
                "would skip the authors' %%.10f Gram -> rank-12 handoff.  Run:\n"
                "  1. KISS_FAITHFUL=1 <this binary> 12 841 35000 SEED SEEDFILE\n"
                "  2. python3 kissing/lib/prepare_841_polish.py CANDIDATE \\\n"
                "       --gram-out GRAM.txt --coords-out PREPARED.txt\n"
                "  3. KISS_FAITHFUL=1 KISS_ADAM_POLISH_ONLY=1 <this binary> "
                "12 841 35000 SEED PREPARED.txt\n");
            return 1;
        }
        if(!adam_polish_only && steps!=35000){
            fprintf(stderr,"KISS_FAITHFUL requires exactly 35000 search steps (got %ld)\n",steps);
            return 1;
        }
        if(argc<=5){
            fprintf(stderr,"KISS_FAITHFUL requires a seed file containing the 840-point core\n");
            return 1;
        }
    }
    { const char*e=getenv("KISS_LOSS"); if(e && !strcmp(e,"ip")) loss_ip=1; }
    if(faithful_mode && loss_ip){
        fprintf(stderr,"KISS_FAITHFUL requires the authors' Riesz loss; unset KISS_LOSS=ip\n");
        return 1;
    }
    { const char*e=getenv("KISS_M"); if(e){ int m=atoi(e); if(m>=3 && m<=MEMMAX) MEM=m; } }
    int inner_cap=16; /* inexact continuation; large caps walk into worse basins */
    { const char*e=getenv("KISS_INNER"); if(e && atoi(e)>0) inner_cap=atoi(e); }
    int do_polish=1;
    { const char*e=getenv("KISS_POLISH"); if(e) do_polish=atoi(e); }
    { const char*e=getenv("KISS_ADAM_EPS"); if(e && atof(e)>0) adam_eps=atof(e); }
    double penalty_target=-1.0;
    { const char*e=getenv("KISS_PENALTY_TARGET"); if(e && atof(e)>=0.5) penalty_target=atof(e); }
    do_profile = env_flag("KISS_PROFILE");

    size_t Nn=(size_t)N*n, NN=(size_t)N*N;
    Gram=malloc(sizeof(double)*NN);
    Cmat=malloc(sizeof(double)*NN);
    GX=malloc(sizeof(double)*Nn);
    rowsum=malloc(sizeof(double)*N);
    Smem=malloc(sizeof(double)*(size_t)MEM*Nn);
    Ymem=malloc(sizeof(double)*(size_t)MEM*Nn);
    double *X=malloc(sizeof(double)*Nn), *G=malloc(sizeof(double)*Nn),
           *B=malloc(sizeof(double)*Nn), *Y=malloc(sizeof(double)*Nn),
           *GY=malloc(sizeof(double)*Nn), *P=malloc(sizeof(double)*Nn);
    unsigned faithful_extra_index=0;
    int faithful_extra_randomized=0;
    double *AdamM=solver==SOLVER_ADAM ? calloc(Nn,sizeof(double)) : NULL;
    double *AdamV=solver==SOLVER_ADAM ? calloc(Nn,sizeof(double)) : NULL;
    double *AdamNorm=solver==SOLVER_ADAM ? malloc(sizeof(double)*(size_t)N) : NULL;
    if(!Gram||!Cmat||!GX||!rowsum||!Smem||!Ymem||
       !X||!G||!B||!Y||!GY||!P){ fprintf(stderr,"oom\n"); return 1; }
    if(solver==SOLVER_ADAM && (!AdamM||!AdamV||!AdamNorm)){ fprintf(stderr,"oom\n"); return 1; }

    for(size_t i=0;i<Nn;i++) X[i]=nrand();
    if(argc>5){ FILE*f=fopen(argv[5],"r"); if(!f){perror("seed");return 1;}
        size_t c=0; char line[65536];
        while(c<Nn && fgets(line,sizeof line,f)){
            if(line[0]=='#') continue;
            char *p=line;
            while(c<Nn){
                char *end; double value=strtod(p,&end);
                if(end==p) break;
                X[c++]=value; p=end;
            }
        }
        fclose(f);
        if(c!=Nn){ fprintf(stderr,"seed: expected %zu coordinates, read %zu\n",Nn,c); return 1; }
        if(faithful_mode && !adam_polish_only){
            /* The authors copy C840 exactly and randomise only the 841st
             * point.  The input's final row is deliberately ignored. */
            const double scale=1.0/sqrt((double)n);
            /* KISS_FAITHFUL_EXTRA pins the 841st hypercube vertex so a run can
             * be replayed exactly, or a specific extra reproduced from a
             * sidecar written by o4_breadth.make_seed.  Without it the index is
             * drawn from the C RNG as the authors do. */
            const char *pin=getenv("KISS_FAITHFUL_EXTRA");
            int pinned=0;
            if(pin && *pin){
                char *end; unsigned long v=strtoul(pin,&end,0);
                if(*end || v>=(1UL<<n)){
                    fprintf(stderr,"KISS_FAITHFUL_EXTRA=%s is not an index in [0,%lu)\n",
                            pin,(unsigned long)1UL<<n);
                    return 1;
                }
                faithful_extra_index=(unsigned)v; pinned=1;
            }
            for(int k=0;k<n;k++){
                unsigned bit = pinned ? ((faithful_extra_index>>k)&1u)
                                      : (unsigned)(uraw()&1ULL);
                if(bit && !pinned) faithful_extra_index|=1u<<k;
                X[(size_t)(N-1)*n+k]=(bit?1.0:-1.0)*scale;
            }
            faithful_extra_randomized=1;
            fprintf(stderr,"faithful init: exact seed core + %s hypercube extra "
                           "index=%u; no jitter\n",
                           pinned?"pinned":"uniform-random", faithful_extra_index);
        }else if(!faithful_mode){
            double jit=0.03; { const char*e=getenv("KISS_JIT"); if(e) jit=atof(e); }
            if(jit>0) for(size_t t=0;t<Nn;t++) X[t]+=jit*nrand();
        }
    }
    /* In faithful search mode, leave the imported 840 core untouched.  Raw
     * Adam evaluates a normalized view, matching the authors' torch code.
     * Polish-only mode starts from a normalized candidate as polish_841.py
     * does; all legacy paths retain their historical normalization. */
    if(!faithful_mode || adam_polish_only) normalize(X);
    double mx0, f0=engrad(X,NULL,1.0,&mx0);
    double best=mx0; memcpy(B,X,sizeof(double)*Nn);
    if(faithful_mode){
        if(f0<0 || !isfinite(f0) || !isfinite(mx0)) faithful_breakdown("initial loss");
        faithful_guard("initial state","X",X,Nn);
        faithful_rows_guard("initial state","X",X);
        if(faithful_failure){
            fprintf(stderr,"faithful run aborted; no candidate was serialized\n");
            return 2;
        }
    }
    const char *solver_name=solver==SOLVER_LBFGS ? "lbfgs" : solver==SOLVER_GD ? "gd" : "adam";
    fprintf(stderr,"start max=%.17g  solver=%s  loss=%s  threads=%d  N=%d n=%d  openblas_threads=1 faithful=%d\n",
            best, solver_name, loss_ip?"ip":"riesz", nthreads, N, n, faithful_mode);

    double s0=0.25, smul=1.12, smax=60000.0;
    if(loss_ip){ s0=2.0; smul=1.18; smax=8000.0; }
    { const char*e;
      if((e=getenv("KISS_S0"))) s0=atof(e);
      if((e=getenv("KISS_SMUL"))) smul=atof(e);
      if((e=getenv("KISS_SMAX"))) smax=atof(e); }
    int nstage=1; for(double q=s0;q<smax;q*=smul) nstage++;
    long per = steps/nstage > 40 ? steps/nstage : 40;
    double t_run=wall();
    int feasible=0;

    int adam_polish=env_flag("KISS_ADAM_POLISH") || adam_polish_only;
    int penalty_only=env_flag("KISS_PENALTY_ONLY");
    if(faithful_mode) fprintf(stderr,"faithful plan: polish=%d polish_only=%d penalty_only=%d do_polish=%d\n",
                             adam_polish,adam_polish_only,penalty_only,do_polish);
    int adam_raw=faithful_mode || env_flag("KISS_ADAM_RAW");
    /* The Adam polish block below requires solver==SOLVER_ADAM.  Without this
     * check, KISS_ADAM_POLISH_ONLY=1 KISS_SOLVER=gd silently ran a full GD
     * search and no polish at all. */
    if(adam_polish_only && solver!=SOLVER_ADAM){
        fprintf(stderr,"KISS_ADAM_POLISH_ONLY requires KISS_SOLVER=adam\n");
        return 1;
    }
    if(adam_polish && solver!=SOLVER_ADAM){
        fprintf(stderr,"KISS_ADAM_POLISH requires KISS_SOLVER=adam\n");
        return 1;
    }
    int adam_base_start=0;
    int adam_base_end=13;
    { const char*e=getenv("KISS_ADAM_BASE_START"); if(e && atoi(e)>=0 && atoi(e)<13) adam_base_start=atoi(e); }
    { const char*e=getenv("KISS_ADAM_BASE_END"); if(e && atoi(e)>0 && atoi(e)<=13) adam_base_end=atoi(e); }
    if(faithful_mode && adam_base_start!=0){
        fprintf(stderr,"KISS_FAITHFUL rejects KISS_ADAM_BASE_START=%d: later-stage starts reset Adam moments\n",
                adam_base_start);
        return 1;
    }
    long adam_search_updates=0;
    if(solver==SOLVER_ADAM && !adam_polish_only && !penalty_only){
        static const double adam_s[]={8,16,32,64,128,256,512,1024,2048,4096,10000,20000,40000};
        static const long adam_base[]={1000,1000,1000,2000,2000,2000,2000,4000,4000,4000,4000,4000,4000};
        static const double adam_lr[]={.005,.003,.002,.001,.0005,.0002,.0001,.00005,.00001,.00001,.000005,.000001,.000001};
        const long published_total=35000;
        double b1pow=1.0,b2pow=1.0;
        for(size_t stage=(size_t)adam_base_start;
            stage<sizeof(adam_s)/sizeof(adam_s[0]) && stage<(size_t)adam_base_end;
            stage++){
            long inner=faithful_mode ? adam_base[stage]
                                     : (long)llround((double)adam_base[stage]*steps/published_total);
            if(inner<1) inner=1;
            int rc;
            if(adam_raw)
                rc=adam_raw_stage(X,G,Y,B,AdamM,AdamV,AdamNorm,
                                  adam_s[stage],inner,adam_lr[stage],
                                  &adam_search_updates,&b1pow,&b2pow,&best);
            else
                rc=adam_stage(X,G,B,AdamM,AdamV,adam_s[stage],inner,adam_lr[stage],
                              &adam_search_updates,&b1pow,&b2pow,&best);
            if(rc<0){ fprintf(stderr,"numerical breakdown at s=%.0f\n",adam_s[stage]); break; }
            fprintf(stderr,"s=%.4g  max=%.12g  nfev=%ld  t=%.1fs\n",
                    adam_s[stage],best,n_engrad,wall()-t_run);
            if(!faithful_mode && (rc==1 || best<=0.5+1e-13)){
                fprintf(stderr,"FEASIBLE s=%.0f max=%.17g\n",adam_s[stage],best);
                feasible=1; break;
            }
        }
    }else if(solver!=SOLVER_ADAM && !penalty_only){
        for(double s=s0; s<smax; s*=smul){
            long inner = solver==SOLVER_LBFGS && per>inner_cap ? inner_cap : per;
            int rc;
            if(solver==SOLVER_LBFGS) rc=lbfgs_stage(X,G,B,Y,GY,P,s,inner,&best);
            else rc=gd_stage(X,G,B,Y,s,per,&best);
            if(rc<0){ fprintf(stderr,"numerical breakdown at s=%.0f\n",s); break; }
            double now=wall()-t_run;
            fprintf(stderr,"s=%.4g  max=%.12g  nfev=%ld  t=%.1fs\n", s, best, n_engrad, now);
            if(rc==1 || best<=0.5+1e-13){
                fprintf(stderr,"FEASIBLE s=%.0f max=%.17g\n",s,best);
                feasible=1; break;
            }
        }
    }

    long polish_updates=0;
    int polish_ran=0;
    if(solver==SOLVER_ADAM && adam_polish && !penalty_only &&
       !feasible && do_polish && (faithful_mode || best>0.5)){
        /* Ultra-high-exponent polishing schedule from the authors' published
         * polish_841.py.  It starts fresh Adam moments from the best candidate.
         * Environment controls make checkpointed, bounded calibration runs
         * reproducible without changing the fixed N=841 seed benchmark. */
        static const double polish_s[]={10240000,20240000,40240000,80240000,160240000};
        static const double polish_lr[]={5e-10,2e-10,1e-10,5e-11,2e-11};
        long polish_steps=150000;
        int polish_stages=5;
        double polish_lr_scale=1.0;
        { const char*e=getenv("KISS_ADAM_POLISH_STEPS"); if(e && atol(e)>0) polish_steps=atol(e); }
        { const char*e=getenv("KISS_ADAM_POLISH_STAGES"); if(e && atoi(e)>0 && atoi(e)<=5) polish_stages=atoi(e); }
        { const char*e=getenv("KISS_ADAM_POLISH_LR_SCALE"); if(e && atof(e)>0) polish_lr_scale=atof(e); }
        fprintf(stderr,"adam polish from max=%.12g  steps/stage=%ld  stages=%d  lr_scale=%.4g\n",
                best,polish_steps,polish_stages,polish_lr_scale);
        memcpy(X,B,sizeof(double)*Nn);
        memset(AdamM,0,sizeof(double)*Nn); memset(AdamV,0,sizeof(double)*Nn);
        long ait=0; double b1pow=1.0,b2pow=1.0;
        polish_ran=1;
        double saved_pair_r2_floor=pair_r2_floor;
        if(faithful_mode) pair_r2_floor=1e-14; /* polish_841.py clamp */
        for(int stage=0;stage<polish_stages;stage++){
            /* polish_lr[] already holds the authors' *effective* rates, i.e.
             * their schedule {5e-9 .. 2e-10} after polish_841.py's own
             * group["lr"]=lr/10.  Dividing again here made the advertised
             * faithful polish ten times smaller than the source. */
            double stage_lr=polish_lr[stage]*polish_lr_scale;
            int rc=adam_stage(X,G,B,AdamM,AdamV,polish_s[stage],polish_steps,
                              stage_lr,
                              &ait,&b1pow,&b2pow,&best);
            if(rc<0){ fprintf(stderr,"numerical breakdown in Adam polish at s=%.0f\n",polish_s[stage]); break; }
            fprintf(stderr,"  polish s=%.4g  max=%.12g  updates=%ld  nfev=%ld  t=%.1fs\n",
                    polish_s[stage],best,ait,n_engrad,wall()-t_run);
            if(!faithful_mode && (rc==1 || best<=0.5+1e-13)){
                fprintf(stderr,"FEASIBLE Adam polish max=%.17g\n",best);
                feasible=1; break;
            }
        }
        pair_r2_floor=saved_pair_r2_floor;
        polish_updates=ait;
    }

    if(!faithful_mode &&
       (solver!=SOLVER_ADAM || penalty_only || !adam_polish) &&
       !feasible && do_polish && best>0.5){
        fprintf(stderr,"penalty polish from max=%.12g\n", best);
        memcpy(X,B,sizeof(double)*Nn);
        double t=penalty_target>=0.5 ? penalty_target : best;
        double last=best; int noimp=0;
        for(int r=0;r<20 && best>0.5+1e-13;r++){
            int rc=lbfgs_pen_stage(X,G,B,Y,GY,P,t, per>80?per:80, &best);
            if(penalty_target<0.5) memcpy(X,B,sizeof(double)*Nn);
            double mx, E=engrad_pen(X,NULL,t,&mx);
            fprintf(stderr,"  polish t=%.12g  E=%.3e  max=%.12g\n", t, E, best);
            if(rc==1 || best<=0.5+1e-13){
                fprintf(stderr,"FEASIBLE polish max=%.17g\n",best);
                feasible=1; break;
            }
            if(last-best<1e-10) noimp++; else noimp=0;
            last=best;
            if(noimp>=4) break;
            if(penalty_target>=0.5){
                t=penalty_target;
            }else if(E<1e-18){
                double gap=t-0.5;
                if(gap<=1e-15) break;
                t -= gap*0.2 + 1e-12; if(t<0.5) t=0.5;
            }else{
                t = 0.7*t + 0.3*best;
            }
        }
    }

    if(faithful_mode){
        if(!isfinite(best) || !faithful_guard("final state","B",B,Nn) ||
           !faithful_rows_guard("final state","B",B)) faithful_failure=1;
        if(faithful_failure){
            fprintf(stderr,"faithful run aborted; no candidate was serialized\n");
            return 2;
        }
    }
    if(do_profile){
        fprintf(stderr,"profile nfev=%ld  gemm=%.3fs  pairs=%.3fs  CX=%.3fs\n",
                n_engrad, t_gemm, t_pairs, t_cx);
    }
    /* In faithful mode `best` is deliberately the final state of the last
     * macro, not the best transient; label it accordingly. */
    printf("n=%d N=%d %s=%.17g feasible=%d\n",n,N,
           faithful_mode ? "final_max_inner" : "best_max_inner",
           best,best<=0.5+1e-12);
    char fn[512];
    if(argc>5) snprintf(fn,sizeof fn,"%s.riesz.s%s.out",argv[5],argv[4]);
    else snprintf(fn,sizeof fn,"riesz_%s.out",argv[4]);
    FILE*f=fopen(fn,"w");
    if(!f){ perror(fn); return 1; }
    fprintf(f,"# n=%d N=%d max_inner=%.17g\n",n,N,best);
    /* Provenance for every run, not just faithful ones: candidate files from
     * different solvers were otherwise byte-format identical. */
    fprintf(f,"# solver=%s loss=%s steps=%ld seed=%s polish=%d\n",
            solver_name, loss_ip?"ip":"riesz", steps, argv[4], do_polish);
    if(faithful_mode){
        /* raw_adam describes the *search*: the polish stages use adam_stage
         * (retract every step), matching the authors' polish_841.py, so a
         * polish-only run has no raw-Adam phase at all.
         * gram_handoff=0 always: the authors' %.10f Gram -> rank-12
         * eigendecomposition step is external (prepare_841_polish.py), so this
         * process never performed it.  A polish-only run may well have been
         * handed a prepared input; C cannot tell, and does not claim to. */
        int search_ran = faithful_extra_randomized;
        if(search_ran)
            fprintf(f,"# faithful=1 raw_adam=1 jitter=0 search_stage_start=%d "
                      "search_stage_end=%d search_updates=%ld full_schedule=%d "
                      "extra_mode=%s extra_index=%u",
                      adam_base_start,adam_base_end,adam_search_updates,
                      adam_base_start==0 && adam_base_end==13 && adam_search_updates==35000,
                      getenv("KISS_FAITHFUL_EXTRA") && *getenv("KISS_FAITHFUL_EXTRA")
                          ? "pinned" : "uniform-random",
                      faithful_extra_index);
        else
            fprintf(f,"# faithful=1 raw_adam=0 jitter=0 search_stage_start=none "
                      "search_stage_end=none search_updates=0 full_schedule=0 "
                      "extra_mode=preserved-input extra_index=none");
        fprintf(f," polish=%d polish_updates=%ld gram_handoff=0\n",
                polish_ran, polish_updates);
    }
    for(int i=0;i<N;i++){ for(int k=0;k<n;k++) fprintf(f,"%.17g ",B[(size_t)i*n+k]); fputc('\n',f); }
    fclose(f); return 0;
}
