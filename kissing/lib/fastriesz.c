/* fastriesz.c -- many-start Riesz-energy continuation for spherical codes.
 *
 * Same objective as riesz.c -- L = log sum_{i<j} r_ij^{-s} on the unit sphere,
 * minimised by Adam through the Takhanov-Assylbekov-Yun exponent / learning
 * rate schedule -- but engineered for *throughput*: one candidate per process,
 * many seeds per process, all cores busy with independent candidates.
 *
 *   - Power-of-two exponents (8 .. 4096) evaluate r^{-s} relative to the
 *     closest pair as repeated squaring of w = r2min/r2: no log/exp, and the
 *     pair loop vectorises.  10000/20000/40000 use exp(s/2 log w) on the few
 *     pairs that survive the cutoff.
 *   - At s >= --list-s only pairs with w^{s/2} > e^{-cut} contribute.  A Verlet
 *     neighbour list (built from one dsyrk, padded by --skin in chord length)
 *     holds them; it is rebuilt whenever 2*max displacement + growth of the
 *     cutoff radius exceeds the skin, so no contributing pair is ever missed.
 *     --check K compares against the full evaluation every K steps.
 *   - Denormals are flushed to zero; they only arise from terms < 1e-300.
 *
 * NEVER build this with -ffast-math or -Ofast: the NaN guards matter (see
 * HANDOFF_optimizer.md, trap 1).  The pair loop vectorises through
 * `#pragma omp simd` (-fopenmp-simd), which needs no unsafe math flags.
 *
 * usage: fastriesz n N INIT OUTDIR [options]
 *   INIT        coordinates, one point per row ('#' comments ignored), with
 *               N rows, or N-1 rows plus an --extra row; or the word 'random'
 *   --seeds A B         run seeds A .. B-1                     (default 0 1)
 *   --adam raw|manifold raw: authors' unconstrained parameters with a
 *                       normalised loss view; manifold: retract every step
 *                                                              (default raw)
 *   --scale F           multiply every stage length by F       (default 1)
 *   --stages K          run only the first K of the 13 stages  (default 13)
 *   --start-stage K     begin at stage K (0-based) with fresh Adam moments,
 *                       e.g. to branch a saved intermediate state  (default 0)
 *   --save-stage K      also write the state reached after stage K (1-based)
 *   --screen K:T        abort a seed if max-IP after stage K exceeds T
 *                       (may be repeated)
 *   --screen-anti K:A   abort a seed with fewer than A near-antipodal pairs
 *                       (cosine < -0.99) after stage K.  For the dim-12
 *                       calibration this separates the two outcomes of the
 *                       first stage (core holds / core melts) cleanly.
 *   --extra hypercube|gauss|file   row N: uniform random (+-1)^n/sqrt(n),
 *                       Gaussian, or the file's own row        (default hypercube)
 *   --jit J             Gaussian jitter added to every row     (default 0)
 *   --keep T            write coordinates when best max-IP < T (default 0.505)
 *   --save-all          write every finished candidate
 *   --cut C             neglect terms below e^-C of the largest (default 40)
 *   --skin D            Verlet skin, chord length              (default 0.03)
 *   --list-s S          neighbour lists for s >= S              (default 256)
 *   --lr-scale F        multiply every learning rate by F      (default 1)
 *   --check K           every K steps compare list vs full evaluation
 *   --dump-grad S       print L and dL/dz at the (seeded) start for exponent S
 *
 * One summary line per seed goes to stdout.  The reported maxima are this
 * program's own float evaluation: recompute from the written file, and use
 * verify_exact.py / certify_float.py before claiming anything.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#ifndef FASTRIESZ_NO_CBLAS_INCLUDE
#include <cblas.h>
#endif

static int n, N;
static size_t Nn;
static double *Z, *P, *Gr, *Cm, *GZ, *rowsum, *grad, *M1, *M2, *norms, *Zb, *Best, *gl;
static int *lj, *rstart;
static size_t lcount, lcap;
static double list_D = 0, list_s = -1;
static long n_rebuild = 0, n_list_eval = 0, n_full_eval = 0;
static double list_pairs_sum = 0;

static double CUT = 40.0, SKIN = 0.03, LIST_S = 256.0, LR_SCALE = 1.0;
static long CHECK = 0;
static double DUMP_S = 0; /* --dump-grad S: print L and gradient at the start, exit */
static const double R2FLOOR = 1e-14;
static const double ADAM_EPS = 1e-8;

static uint64_t rng_state;
static uint64_t splitmix(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double urand(void) { return (splitmix() >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) {
    double u = urand() + 1e-300, v = urand();
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * v);
}
static double wall(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int pow2_index(double h) { /* k with h == 2^k, else -1 */
    if (!(h >= 1.0) || h > 1e18) return -1;
    double x = h; int k = 0;
    while (x > 1.0) { if (fmod(x, 2.0) != 0.0) return -1; x /= 2.0; k++; }
    return x == 1.0 ? k : -1;
}

static void normalize_rows(double *X) {
    for (int i = 0; i < N; i++) {
        double *x = X + (size_t)i * n, q = 0;
        for (int k = 0; k < n; k++) q += x[k] * x[k];
        q = q > 0 ? 1.0 / sqrt(q) : 1.0;
        for (int k = 0; k < n; k++) x[k] *= q;
    }
}

static long antipodal_pairs(const double *X) { /* pairs with cosine < -0.99 */
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, N, n, 1.0, X, n, 0.0, Gr, N);
    long c = 0;
    for (int i = 0; i < N; i++) {
        const double *gi = Gr + (size_t)i * N;
        for (int j = i + 1; j < N; j++) c += gi[j] < -0.99;
    }
    return c;
}

static double gram_max(const double *X) {
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, N, n, 1.0, X, n, 0.0, Gr, N);
    double m = -2;
    for (int i = 0; i < N; i++) {
        const double *gi = Gr + (size_t)i * N;
#pragma omp simd reduction(max:m)
        for (int j = i + 1; j < N; j++) m = gi[j] > m ? gi[j] : m;
    }
    return m;
}

/* e <- w^(2^K) */
#define SQ1(e) e *= e;
#define SQ2(e) SQ1(e) SQ1(e)
#define SQ3(e) SQ2(e) SQ1(e)
#define SQ4(e) SQ3(e) SQ1(e)
#define SQ5(e) SQ4(e) SQ1(e)
#define SQ6(e) SQ5(e) SQ1(e)
#define SQ7(e) SQ6(e) SQ1(e)
#define SQ8(e) SQ7(e) SQ1(e)
#define SQ9(e) SQ8(e) SQ1(e)
#define SQ10(e) SQ9(e) SQ1(e)
#define SQ11(e) SQ10(e) SQ1(e)
#define SQ12(e) SQ11(e) SQ1(e)

#define FULL_LOOP(SQK)                                                    \
    _Pragma("omp simd reduction(+:E,rs)")                                 \
    for (int j = i + 1; j < N; j++) {                                     \
        double r2 = 2.0 - 2.0 * gi[j];                                    \
        r2 = r2 < R2FLOOR ? R2FLOOR : r2;                                 \
        double w = r2min / r2, e = w;                                     \
        SQK(e)                                                            \
        E += e;                                                           \
        double c = cs * e * w;                                            \
        ci[j] = c; rs += c; rowsum[j] += c;                               \
    }

/* Full evaluation.  Returns L = log sum r^{-s}; grad = dL/dz (Euclidean,
 * not yet projected).  NAN on breakdown. */
static double eval_full(const double *X, double *G, double s, double *mx) {
    n_full_eval++;
    double gmax = gram_max(X);
    *mx = gmax;
    if (!(gmax > -1.5) || !isfinite(gmax)) return NAN;
    double r2min = 2 - 2 * gmax; if (r2min < R2FLOOR) r2min = R2FLOOR;
    if (!G) return 0.0;
    double halfs = 0.5 * s, cs = s / r2min, E = 0;
    int K = pow2_index(halfs);
    memset(rowsum, 0, sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        const double *gi = Gr + (size_t)i * N;
        double *ci = Cm + (size_t)i * N, rs = 0;
        ci[i] = 0;
        switch (K) {
        case 1: FULL_LOOP(SQ1) break;
        case 2: FULL_LOOP(SQ2) break;
        case 3: FULL_LOOP(SQ3) break;
        case 4: FULL_LOOP(SQ4) break;
        case 5: FULL_LOOP(SQ5) break;
        case 6: FULL_LOOP(SQ6) break;
        case 7: FULL_LOOP(SQ7) break;
        case 8: FULL_LOOP(SQ8) break;
        case 9: FULL_LOOP(SQ9) break;
        case 10: FULL_LOOP(SQ10) break;
        case 11: FULL_LOOP(SQ11) break;
        case 12: FULL_LOOP(SQ12) break;
        default:
            for (int j = i + 1; j < N; j++) {
                double r2 = 2.0 - 2.0 * gi[j];
                r2 = r2 < R2FLOOR ? R2FLOOR : r2;
                double w = r2min / r2, lw = log(w);
                if (-halfs * lw > CUT) { ci[j] = 0; continue; }
                double e = exp(halfs * lw);
                E += e;
                double c = cs * e * w;
                ci[j] = c; rs += c; rowsum[j] += c;
            }
        }
        rowsum[i] += rs;
    }
    if (!(E > 0) || !isfinite(E)) return NAN;
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, N, n, 1.0, Cm, N, X, n, 0.0, GZ, n);
    double invE = 1.0 / E;
    for (int i = 0; i < N; i++) {
        const double *gz = GZ + (size_t)i * n, *x = X + (size_t)i * n;
        double *g = G + (size_t)i * n, r = rowsum[i];
        for (int k = 0; k < n; k++) g[k] = (gz[k] - r * x[k]) * invE;
    }
    return log(E) - halfs * log(r2min);
}

static double cutoff_radius(double r2min, double s) { return sqrt(r2min * exp(2.0 * CUT / s)); }

static void build_list(const double *X, double s) {
    double gmax = gram_max(X);
    double r2min = 2 - 2 * gmax; if (r2min < R2FLOOR) r2min = R2FLOOR;
    list_D = cutoff_radius(r2min, s);
    double Dl = list_D + SKIN, gthr = 1.0 - 0.5 * Dl * Dl;
    size_t c = 0;
    for (int i = 0; i < N; i++) {
        rstart[i] = (int)c;
        const double *gi = Gr + (size_t)i * N;
        for (int j = i + 1; j < N; j++)
            if (gi[j] > gthr) lj[c++] = j;
    }
    rstart[N] = (int)c;
    lcount = c;
    memcpy(Zb, X, sizeof(double) * Nn);
    list_s = s;
    n_rebuild++;
}

static double max_disp(const double *X) {
    double m = 0;
    for (int i = 0; i < N; i++) {
        const double *x = X + (size_t)i * n, *y = Zb + (size_t)i * n;
        double q = 0;
        for (int k = 0; k < n; k++) { double d = x[k] - y[k]; q += d * d; }
        if (q > m) m = q;
    }
    return sqrt(m);
}

static double eval_list(const double *X, double *G, double s, double *mx) {
    if (list_s != s) build_list(X, s);
    double disp = max_disp(X);
    if (2 * disp > SKIN) { build_list(X, s); disp = 0; }
    for (int attempt = 0; attempt < 2; attempt++) {
        double gmax = -2;
        for (int i = 0; i < N; i++) {
            const double *xi = X + (size_t)i * n;
            for (int p = rstart[i]; p < rstart[i + 1]; p++) {
                const double *xj = X + (size_t)lj[p] * n;
                double g = 0;
                for (int k = 0; k < n; k++) g += xi[k] * xj[k];
                gl[p] = g;
                if (g > gmax) gmax = g;
            }
        }
        if (!(gmax > -1.5) || !isfinite(gmax)) { *mx = gmax; return NAN; }
        double r2min = 2 - 2 * gmax; if (r2min < R2FLOOR) r2min = R2FLOOR;
        if (attempt == 0 && 2 * disp + (cutoff_radius(r2min, s) - list_D) > SKIN) {
            build_list(X, s); disp = 0;
            continue;
        }
        *mx = gmax;
        n_list_eval++;
        list_pairs_sum += (double)lcount;
        if (!G) return 0.0;
        double halfs = 0.5 * s, cs = s / r2min, E = 0;
        int K = pow2_index(halfs);
        memset(G, 0, sizeof(double) * Nn);
        for (int i = 0; i < N; i++) {
            const double *xi = X + (size_t)i * n;
            double *Gi = G + (size_t)i * n;
            for (int p = rstart[i]; p < rstart[i + 1]; p++) {
                double r2 = 2.0 - 2.0 * gl[p];
                r2 = r2 < R2FLOOR ? R2FLOOR : r2;
                double w = r2min / r2, e;
                if (K >= 0) { e = w; for (int q = 0; q < K; q++) e *= e; }
                else {
                    double lw = log(w);
                    if (-halfs * lw > CUT) continue;
                    e = exp(halfs * lw);
                }
                E += e;
                double c = cs * e * w;
                int j = lj[p];
                const double *xj = X + (size_t)j * n;
                double *Gj = G + (size_t)j * n;
                for (int k = 0; k < n; k++) {
                    double d = xj[k] - xi[k];
                    Gi[k] += c * d;
                    Gj[k] -= c * d;
                }
            }
        }
        if (!(E > 0) || !isfinite(E)) return NAN;
        double invE = 1.0 / E;
        for (size_t t = 0; t < Nn; t++) G[t] *= invE;
        return log(E) - halfs * log(r2min);
    }
    return NAN; /* unreachable */
}

static double eval(const double *X, double *G, double s, double *mx) {
    if (s >= LIST_S) return eval_list(X, G, s, mx);
    return eval_full(X, G, s, mx);
}

static void project_tangent(const double *X, double *G) {
    for (int i = 0; i < N; i++) {
        const double *x = X + (size_t)i * n; double *g = G + (size_t)i * n, d = 0;
        for (int k = 0; k < n; k++) d += g[k] * x[k];
        for (int k = 0; k < n; k++) g[k] -= d * x[k];
    }
}

static void check_against_full(const double *X, double s, long it) {
    double *G1 = malloc(sizeof(double) * Nn), *G2 = malloc(sizeof(double) * Nn), m1, m2;
    double f1 = eval(X, G1, s, &m1);
    double f2 = eval_full(X, G2, s, &m2);
    double dn = 0, gn = 0;
    for (size_t t = 0; t < Nn; t++) { dn += (G1[t] - G2[t]) * (G1[t] - G2[t]); gn += G2[t] * G2[t]; }
    fprintf(stderr, "check s=%g it=%ld  L %.15g vs %.15g  max %.15g vs %.15g  |dG|/|G| %.2e  list=%zu\n",
            s, it, f1, f2, m1, m2, sqrt(dn / (gn > 0 ? gn : 1)), lcount);
    free(G1); free(G2);
}

typedef struct { double s; long it; double lr; } Stage;
static const Stage SCHED[13] = {
    {8, 1000, .005}, {16, 1000, .003}, {32, 1000, .002}, {64, 2000, .001},
    {128, 2000, .0005}, {256, 2000, .0002}, {512, 2000, .0001}, {1024, 4000, .00005},
    {2048, 4000, .00001}, {4096, 4000, .00001}, {10000, 4000, .000005},
    {20000, 4000, .000001}, {40000, 4000, .000001}};

static double best_max;
static long adam_t;
static double b1p, b2p;

/* One stage of Adam.  raw=1: parameters P, loss on Z=P/|P|.  raw=0: Z is the
 * parameter and is retracted after every step.  Returns final max-IP or NAN. */
static double run_stage(double s, long iters, double lr, int raw) {
    const double b1 = 0.9, b2 = 0.999;
    double mx = 0;
    for (long it = 0; it < iters; it++) {
        if (raw) {
            for (int i = 0; i < N; i++) {
                const double *p = P + (size_t)i * n; double *z = Z + (size_t)i * n, q = 0;
                for (int k = 0; k < n; k++) q += p[k] * p[k];
                double r = sqrt(q); if (!(r > 1e-12)) r = 1e-12;
                norms[i] = r;
                for (int k = 0; k < n; k++) z[k] = p[k] / r;
            }
        }
        if (CHECK > 0 && it % CHECK == 0) check_against_full(Z, s, it);
        double f = eval(Z, grad, s, &mx);
        if (!isfinite(f) || !isfinite(mx)) return NAN;
        if (mx < best_max) { best_max = mx; memcpy(Best, Z, sizeof(double) * Nn); }
        project_tangent(Z, grad);
        if (raw)
            for (int i = 0; i < N; i++) {
                double inv = 1.0 / norms[i], *g = grad + (size_t)i * n;
                for (int k = 0; k < n; k++) g[k] *= inv;
            }
        adam_t++; b1p *= b1; b2p *= b2;
        double c1 = 1.0 / (1.0 - b1p), c2 = 1.0 / (1.0 - b2p);
        double *par = raw ? P : Z;
        for (size_t t = 0; t < Nn; t++) {
            double g = grad[t];
            M1[t] = b1 * M1[t] + (1 - b1) * g;
            M2[t] = b2 * M2[t] + (1 - b2) * g * g;
            par[t] -= lr * (M1[t] * c1) / (sqrt(M2[t] * c2) + ADAM_EPS);
        }
        if (!raw) normalize_rows(Z);
    }
    if (raw) { memcpy(Z, P, sizeof(double) * Nn); normalize_rows(Z); }
    mx = gram_max(Z);
    if (!isfinite(mx)) return NAN;
    if (mx < best_max) { best_max = mx; memcpy(Best, Z, sizeof(double) * Nn); }
    return mx;
}

static void write_points(const char *path, const double *X, const char *note) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    double m = gram_max(X);
    fprintf(f, "# n=%d N=%d max_inner=%.17g %s\n", n, N, m, note);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < n; k++) fprintf(f, "%.17g%c", X[(size_t)i * n + k], k + 1 < n ? ' ' : '\n');
    }
    fclose(f);
}

static double *read_points(const char *path, int *rows) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return NULL; }
    size_t cap = (size_t)N * n, c = 0;
    double *A = malloc(sizeof(double) * cap);
    char *line = NULL; size_t ln = 0;
    while (getline(&line, &ln, f) > 0) {
        if (line[0] == '#') continue;
        char *p = line, *end;
        int got = 0;
        for (;;) {
            double v = strtod(p, &end);
            if (end == p) break;
            if (c >= cap) { fprintf(stderr, "%s: more than %d rows\n", path, N); free(A); free(line); fclose(f); return NULL; }
            A[c++] = v; p = end; got++;
        }
        if (got && got != n) { fprintf(stderr, "%s: row with %d coordinates, want %d\n", path, got, n); free(A); free(line); fclose(f); return NULL; }
    }
    free(line); fclose(f);
    *rows = (int)(c / n);
    return A;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: fastriesz n N INIT OUTDIR [options]  (see the header comment)\n");
        return 1;
    }
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    openblas_set_num_threads(1); /* one candidate per process; parallelise over processes */
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    n = atoi(argv[1]); N = atoi(argv[2]);
    const char *init = argv[3], *outdir = argv[4];
    if (n < 2 || N < 2) { fprintf(stderr, "bad n/N\n"); return 1; }
    Nn = (size_t)N * n;
    long seed_a = 0, seed_b = 1; int raw = 1, nstages = 13, save_all = 0;
    int start_stage = 0, save_stage = 0;
    double scale = 1.0, jit = 0.0, keep = 0.505;
    const char *extra = "hypercube";
    int scr_k[16]; double scr_t[16]; int nscr = 0;
    int anti_k[16]; long anti_a[16]; int nanti = 0;
    for (int a = 5; a < argc; a++) {
        const char *o = argv[a];
#define NEXT (a + 1 < argc ? argv[++a] : (fprintf(stderr, "%s needs a value\n", o), exit(1), ""))
        if (!strcmp(o, "--seeds")) { seed_a = atol(NEXT); seed_b = atol(NEXT); }
        else if (!strcmp(o, "--adam")) { const char *v = NEXT; raw = !strcmp(v, "raw"); if (!raw && strcmp(v, "manifold")) { fprintf(stderr, "--adam raw|manifold\n"); return 1; } }
        else if (!strcmp(o, "--scale")) scale = atof(NEXT);
        else if (!strcmp(o, "--stages")) nstages = atoi(NEXT);
        else if (!strcmp(o, "--start-stage")) start_stage = atoi(NEXT);
        else if (!strcmp(o, "--save-stage")) save_stage = atoi(NEXT);
        else if (!strcmp(o, "--screen")) {
            const char *v = NEXT; int k; double t;
            if (nscr >= 16 || sscanf(v, "%d:%lf", &k, &t) != 2) { fprintf(stderr, "--screen K:T\n"); return 1; }
            scr_k[nscr] = k; scr_t[nscr] = t; nscr++;
        }
        else if (!strcmp(o, "--screen-anti")) {
            const char *v = NEXT; int k; long t;
            if (nanti >= 16 || sscanf(v, "%d:%ld", &k, &t) != 2) { fprintf(stderr, "--screen-anti K:A\n"); return 1; }
            anti_k[nanti] = k; anti_a[nanti] = t; nanti++;
        }
        else if (!strcmp(o, "--extra")) extra = NEXT;
        else if (!strcmp(o, "--jit")) jit = atof(NEXT);
        else if (!strcmp(o, "--keep")) keep = atof(NEXT);
        else if (!strcmp(o, "--save-all")) save_all = 1;
        else if (!strcmp(o, "--cut")) CUT = atof(NEXT);
        else if (!strcmp(o, "--skin")) SKIN = atof(NEXT);
        else if (!strcmp(o, "--list-s")) LIST_S = atof(NEXT);
        else if (!strcmp(o, "--lr-scale")) LR_SCALE = atof(NEXT);
        else if (!strcmp(o, "--check")) CHECK = atol(NEXT);
        else if (!strcmp(o, "--dump-grad")) DUMP_S = atof(NEXT);
        else { fprintf(stderr, "unknown option %s\n", o); return 1; }
    }
    if (nstages < 1 || nstages > 13) { fprintf(stderr, "--stages must be 1..13\n"); return 1; }
    if (start_stage < 0 || start_stage >= nstages) { fprintf(stderr, "--start-stage must be in 0..stages-1\n"); return 1; }

    double *init_pts = NULL; int init_rows = 0;
    if (strcmp(init, "random")) {
        init_pts = read_points(init, &init_rows);
        if (!init_pts) return 1;
        if (init_rows != N && init_rows != N - 1) {
            fprintf(stderr, "%s: %d rows, want %d or %d\n", init, init_rows, N - 1, N);
            return 1;
        }
        if (init_rows == N - 1 && !strcmp(extra, "file")) {
            fprintf(stderr, "--extra file needs %d rows\n", N); return 1;
        }
    }

    Z = malloc(sizeof(double) * Nn); P = malloc(sizeof(double) * Nn);
    grad = malloc(sizeof(double) * Nn); GZ = malloc(sizeof(double) * Nn);
    M1 = malloc(sizeof(double) * Nn); M2 = malloc(sizeof(double) * Nn);
    Zb = malloc(sizeof(double) * Nn); Best = malloc(sizeof(double) * Nn);
    norms = malloc(sizeof(double) * N); rowsum = malloc(sizeof(double) * N);
    Gr = malloc(sizeof(double) * (size_t)N * N); Cm = malloc(sizeof(double) * (size_t)N * N);
    lcap = (size_t)N * (N - 1) / 2;
    lj = malloc(sizeof(int) * lcap); gl = malloc(sizeof(double) * lcap);
    rstart = malloc(sizeof(int) * (N + 1));
    if (!Z || !P || !grad || !GZ || !M1 || !M2 || !Zb || !Best || !norms || !rowsum ||
        !Gr || !Cm || !lj || !gl || !rstart) { fprintf(stderr, "oom\n"); return 1; }

    long total_steps = 0;
    for (int st = start_stage; st < nstages; st++) total_steps += (long)llround(SCHED[st].it * scale);
    fprintf(stderr, "fastriesz n=%d N=%d init=%s adam=%s scale=%g stages=%d steps=%ld extra=%s jit=%g cut=%g skin=%g list_s=%g\n",
            n, N, init, raw ? "raw" : "manifold", scale, nstages, total_steps, extra, jit, CUT, SKIN, LIST_S);

    for (long seed = seed_a; seed < seed_b; seed++) {
        rng_state = 0x5DEECE66DULL ^ ((uint64_t)seed * 0x2545F4914F6CDD1DULL);
        double t0 = wall();
        unsigned long extra_index = 0;
        if (!init_pts) {
            for (size_t t = 0; t < Nn; t++) P[t] = nrand();
        } else {
            memcpy(P, init_pts, sizeof(double) * (size_t)init_rows * n);
            if (init_rows == N - 1 || strcmp(extra, "file")) {
                double *x = P + (size_t)(N - 1) * n;
                if (!strcmp(extra, "gauss")) {
                    for (int k = 0; k < n; k++) x[k] = nrand();
                } else { /* hypercube */
                    for (int k = 0; k < n; k++) {
                        int bit = (int)(splitmix() & 1);
                        if (bit) extra_index |= 1UL << k;
                        x[k] = (bit ? 1.0 : -1.0) / sqrt((double)n);
                    }
                }
            }
            if (jit > 0) for (size_t t = 0; t < Nn; t++) P[t] += jit * nrand();
        }
        if (!raw) normalize_rows(P);
        memcpy(Z, P, sizeof(double) * Nn);
        normalize_rows(Z);
        memset(M1, 0, sizeof(double) * Nn); memset(M2, 0, sizeof(double) * Nn);
        adam_t = 0; b1p = 1; b2p = 1; best_max = 3; list_s = -1;
        n_rebuild = n_list_eval = n_full_eval = 0; list_pairs_sum = 0;
        double start_max = gram_max(Z);
        if (DUMP_S > 0) {
            double m, f = eval(Z, grad, DUMP_S, &m);
            printf("L=%.17g max=%.17g\n", f, m);
            for (size_t t = 0; t < Nn; t++) printf("%.17g%c", grad[t], (t + 1) % n ? ' ' : '\n');
            return 0;
        }

        double smax[13]; int done = 0, screened = 0, broke = 0;
        long anti_seen = -1;
        for (int st = start_stage; st < nstages; st++) {
            long iters = (long)llround(SCHED[st].it * scale);
            double m = run_stage(SCHED[st].s, iters, SCHED[st].lr * LR_SCALE, raw);
            if (!isfinite(m)) { broke = 1; break; }
            smax[done++] = m;
            if (save_stage == st + 1) {
                char path[4096], note[256];
                snprintf(note, sizeof note, "fastriesz seed=%ld adam=%s scale=%g state_after_stage=%d",
                         seed, raw ? "raw" : "manifold", scale, st + 1);
                snprintf(path, sizeof path, "%s/state_n%d_N%d_s%ld_k%d.txt", outdir, n, N, seed, st + 1);
                if (raw) { memcpy(Z, P, sizeof(double) * Nn); normalize_rows(Z); }
                write_points(path, Z, note);
            }
            for (int q = 0; q < nscr; q++)
                if (scr_k[q] == st + 1 && m > scr_t[q]) screened = 1;
            for (int q = 0; q < nanti; q++)
                if (anti_k[q] == st + 1) {
                    anti_seen = antipodal_pairs(Z);
                    if (anti_seen < anti_a[q]) screened = 1;
                }
            if (screened) break;
        }
        double final = done ? smax[done - 1] : NAN;
        printf("seed=%ld extra=%lu start=%.6f stages=", seed, extra_index, start_max);
        for (int q = 0; q < done; q++) printf("%s%.9f", q ? "," : "", smax[q]);
        printf(" final=%.15f best=%.15f status=%s time=%.1f rebuilds=%ld avg_list=%.0f anti=%ld\n",
               final, best_max, broke ? "breakdown" : screened ? "screened" : "done",
               wall() - t0, n_rebuild, n_list_eval ? list_pairs_sum / n_list_eval : 0.0, anti_seen);
        fflush(stdout);
        if (!broke && (save_all || best_max < keep)) {
            char path[4096], note[256];
            snprintf(note, sizeof note, "fastriesz seed=%ld adam=%s scale=%g stages=%d best", seed,
                     raw ? "raw" : "manifold", scale, done);
            snprintf(path, sizeof path, "%s/cand_n%d_N%d_s%ld.txt", outdir, n, N, seed);
            write_points(path, Best, note);
        }
    }
    return 0;
}
