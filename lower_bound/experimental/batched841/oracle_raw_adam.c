/*
 * Tiny machine-readable oracle for test_parity.py.
 *
 * The source under test is included deliberately: this gives the parity test
 * access to riesz.c's static engrad_riesz implementation without modifying
 * the historical optimizer.  This is a test helper, never a replacement for
 * the riesz executable.  The caller supplies a binary row-major float64 raw
 * state and a text schedule (one ``s steps lr`` tuple per line).  The oracle
 * emits one record per Adam update, including the pre-update loss/gradient and
 * post-update raw coordinates and moments.
 */
#define main riesz_source_main
#include "../../lib/riesz.c"
#undef main

#include <errno.h>

static int read_schedule(const char *path, double **ss_out, long **nn_out,
                         double **lr_out, size_t *count_out){
    FILE *f=fopen(path,"r");
    if(!f){ fprintf(stderr,"schedule: %s: %s\n",path,strerror(errno)); return 0; }
    size_t cap=8, count=0;
    double *ss=malloc(cap*sizeof(*ss)), *lr=malloc(cap*sizeof(*lr));
    long *nn=malloc(cap*sizeof(*nn));
    if(!ss||!nn||!lr){ fclose(f); free(ss); free(nn); free(lr); return 0; }
    char line[256];
    while(fgets(line,sizeof line,f)){
        if(line[0]=='#' || line[0]=='\n' || line[0]=='\0') continue;
        double s=0, rate=0; long steps=0;
        if(sscanf(line," %lf %ld %lf",&s,&steps,&rate)!=3 ||
           !(s>0) || steps<1 || !(rate>0) || !isfinite(s) || !isfinite(rate)){
            fprintf(stderr,"schedule: malformed line: %s",line);
            free(ss); free(nn); free(lr); fclose(f); return 0;
        }
        if(count==cap){
            cap*=2;
            double *nss=realloc(ss,cap*sizeof(*ss));
            long *nnn=realloc(nn,cap*sizeof(*nn));
            double *nlr=realloc(lr,cap*sizeof(*lr));
            if(!nss||!nnn||!nlr){ free(nss); free(nnn); free(nlr); fclose(f); return 0; }
            ss=nss; nn=nnn; lr=nlr;
        }
        ss[count]=s; nn[count]=steps; lr[count]=rate; count++;
    }
    fclose(f);
    if(count==0){ fprintf(stderr,"schedule: no stages\n"); free(ss); free(nn); free(lr); return 0; }
    *ss_out=ss; *nn_out=nn; *lr_out=lr; *count_out=count; return 1;
}

static int read_raw(const char *path, double *x, size_t count){
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"raw: %s: %s\n",path,strerror(errno)); return 0; }
    size_t got=fread(x,sizeof(*x),count,f);
    int ok=got==count;
    if(ok){ int ch=fgetc(f); if(ch!=EOF) ok=0; }
    fclose(f);
    if(!ok) fprintf(stderr,"raw: expected exactly %zu float64 values, read %zu\n",count,got);
    return ok;
}

static void print_values(FILE *out, const char *label, const double *x, size_t count){
    fputs(label,out);
    for(size_t i=0;i<count;i++) fprintf(out," %.17g",x[i]);
    fputc('\n',out);
}

static int valid_rows(const double *x){
    for(int i=0;i<N;i++){
        double q=0; const double *r=x+(size_t)i*n;
        for(int k=0;k<n;k++) q+=r[k]*r[k];
        if(!(q>1e-30) || !isfinite(q)) return 0;
    }
    return 1;
}

int main(int argc, char **argv){
    if(argc<6 || argc>7){
        fprintf(stderr,"usage: %s n N raw.bin schedule.txt output.txt [eps]\n",argv[0]);
        return 2;
    }
    n=atoi(argv[1]); N=atoi(argv[2]);
    if(n<1 || N<2){ fprintf(stderr,"invalid shape\n"); return 2; }
    const size_t Nn=(size_t)N*n, NN=(size_t)N*N;
    const double eps=argc==7 ? atof(argv[6]) : 1e-8;
    if(!(eps>=0) || !isfinite(eps)){ fprintf(stderr,"invalid epsilon\n"); return 2; }
    adam_eps=eps;
    faithful_mode=1; loss_ip=0; pair_r2_floor=1e-12;
    do_profile=0; n_engrad=0;

    double *ss=NULL, *lr=NULL, *X=NULL, *Z=NULL, *G=NULL, *B=NULL, *M1=NULL, *M2=NULL, *norms=NULL;
    long *steps=NULL; size_t nstage=0;
    if(!read_schedule(argv[4],&ss,&steps,&lr,&nstage)) return 2;
    X=malloc(sizeof(*X)*Nn); Z=malloc(sizeof(*Z)*Nn); G=malloc(sizeof(*G)*Nn);
    B=malloc(sizeof(*B)*Nn); norms=malloc(sizeof(*norms)*(size_t)N);
    M1=calloc(Nn,sizeof(*M1)); M2=calloc(Nn,sizeof(*M2));
    Gram=malloc(sizeof(*Gram)*NN); Cmat=malloc(sizeof(*Cmat)*NN);
    GX=malloc(sizeof(*GX)*Nn); rowsum=malloc(sizeof(*rowsum)*N);
    if(!X||!Z||!G||!B||!norms||!M1||!M2||!Gram||!Cmat||!GX||!rowsum){
        fprintf(stderr,"out of memory\n"); return 2;
    }
    if(!read_raw(argv[3],X,Nn) || !finite_block(X,Nn) || !valid_rows(X)){
        fprintf(stderr,"raw state is nonfinite or has a zero row\n"); return 3;
    }

    FILE *out=fopen(argv[5],"w");
    if(!out){ fprintf(stderr,"output: %s: %s\n",argv[5],strerror(errno)); return 2; }
    fprintf(out,"# n=%d N=%d eps=%.17g\n",n,N,eps);

    long global_step=0;
    double b1pow=1.0, b2pow=1.0;
    double best=DBL_MAX;
    for(size_t stage=0; stage<nstage; stage++){
        fprintf(out,"stage %zu %.17g %ld %.17g\n",stage,ss[stage],steps[stage],lr[stage]);
        for(long iter=0; iter<steps[stage]; iter++){
            if(!finite_block(X,Nn) || !valid_rows(X)){ fclose(out); return 3; }
            for(int i=0;i<N;i++){
                const double *x=X+(size_t)i*n; double *z=Z+(size_t)i*n;
                double q=0; for(int k=0;k<n;k++) q+=x[k]*x[k];
                double r=sqrt(q);
                if(!(r>1e-12) || !isfinite(r)){ fclose(out); return 3; }
                for(int k=0;k<n;k++) z[k]=x[k]/r;
            }
            double mx=0, loss=engrad_riesz(Z,G,ss[stage],&mx);
            if(!(loss>=0) || !isfinite(loss) || !isfinite(mx) || !finite_block(G,Nn)){
                fclose(out); return 3;
            }
            /* engrad_riesz removes each row's radial component; the source
             * adam_raw_stage then applies its full tangent projection once
             * more before the raw-view chain rule.  Keep both operations here
             * to record the exact pre-update gradient. */
            project_tangent(Z,G);
            for(int i=0;i<N;i++){
                const double *x=X+(size_t)i*n; double q=0;
                for(int k=0;k<n;k++) q+=x[k]*x[k];
                double r=sqrt(q), inv=1.0/r;
                for(int k=0;k<n;k++) G[(size_t)i*n+k]*=inv;
            }
            /* Invoke the actual source optimizer, rather than copying its
             * Adam update in this helper.  One call per record preserves the
             * moments and bias-correction powers across stages while making
             * every intermediate state observable to the Python checker. */
            int rc=adam_raw_stage(X,G,Z,B,M1,M2,norms,
                                  ss[stage],1,lr[stage],&global_step,
                                  &b1pow,&b2pow,&best);
            if(rc<0){ fclose(out); return 3; }
            fprintf(out,"record %ld %zu %ld %.17g %.17g %.17g\n",
                    global_step,stage,iter,ss[stage],loss,mx);
            print_values(out,"grad",G,Nn);
            print_values(out,"raw",X,Nn);
            print_values(out,"m",M1,Nn);
            print_values(out,"v",M2,Nn);
        }
    }
    fclose(out);
    free(ss); free(steps); free(lr); free(X); free(Z); free(G); free(B); free(norms); free(M1); free(M2);
    free(Gram); free(Cmat); free(GX); free(rowsum);
    return 0;
}
