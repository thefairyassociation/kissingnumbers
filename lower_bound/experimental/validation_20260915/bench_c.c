/* Timing adapter: calls the unchanged C optimizer, with no per-update I/O.
 * Input: exactly 841*12 binary doubles and a text s/steps/lr schedule.
 * Output: stage timings on stdout and final normalized coordinates to a file.
 * Build flags must match the existing Makefile; never use fast-math.
 */
#define main original_riesz_main
#include "../../lib/riesz.c"
#undef main

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: bench_c raw.bin schedule.txt output.txt threads\n");
        return 2;
    }
    n = 12; N = 841;
    faithful_mode = 1; loss_ip = 0; pair_r2_floor = 1e-12;
    adam_eps = 1e-8; do_profile = 0;
    int threads = atoi(argv[4]);
    if (threads < 1) return 2;
    omp_set_num_threads(threads);
    openblas_set_num_threads(1);
    size_t size = (size_t)N*n;
    double *X = malloc(size*sizeof(double)), *G = malloc(size*sizeof(double));
    double *Z = malloc(size*sizeof(double)), *B = malloc(size*sizeof(double));
    double *M = calloc(size,sizeof(double)), *V = calloc(size,sizeof(double));
    double *norms = malloc(N*sizeof(double));
    Gram = malloc((size_t)N*N*sizeof(double));
    Cmat = malloc((size_t)N*N*sizeof(double));
    GX = malloc(size*sizeof(double)); rowsum = malloc(N*sizeof(double));
    if (!X || !G || !Z || !B || !M || !V || !norms || !Gram || !Cmat || !GX || !rowsum) return 2;
    FILE *input = fopen(argv[1], "rb");
    if (!input || fread(X,sizeof(double),size,input) != size || fgetc(input) != EOF) return 2;
    fclose(input);
    FILE *schedule = fopen(argv[2], "r");
    if (!schedule) return 2;
    double exponent, rate, b1 = 1, b2 = 1, best = DBL_MAX;
    long steps, updates = 0; int stage = 0;
    while (fscanf(schedule,"%lf %ld %lf",&exponent,&steps,&rate) == 3) {
        if (!(exponent > 0) || steps < 1 || !(rate > 0)) return 2;
        double start = wall();
        int rc = adam_raw_stage(X,G,Z,B,M,V,norms,exponent,steps,rate,&updates,&b1,&b2,&best);
        double elapsed = wall()-start;
        if (rc < 0) return 3;
        printf("{\"stage\":%d,\"s\":%.17g,\"steps\":%ld,\"seconds\":%.9g,\"max_ip\":%.17g}\n",stage++,exponent,steps,elapsed,best);
        fflush(stdout);
    }
    fclose(schedule);
    if (!stage) return 2;
    FILE *output = fopen(argv[3],"w");
    if (!output) return 2;
    for (int i=0;i<N;i++) {
        for (int j=0;j<n;j++) fprintf(output,"%.17g%c",B[(size_t)i*n+j],j+1==n?'\n':' ');
    }
    if (fclose(output)) return 2;
    free(X); free(G); free(Z); free(B); free(M); free(V); free(norms);
    free(Gram); free(Cmat); free(GX); free(rowsum);
    return 0;
}
