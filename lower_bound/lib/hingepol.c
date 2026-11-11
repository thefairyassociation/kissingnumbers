/* hingepol.c -- exact-target hinge polish for spherical codes.
 *
 * Minimises  E_t(X) = sum_{i<j} max(0, <x_i,x_j> - t)^2  over unit vectors by
 * L-BFGS on the raw coordinates (the loss reads the normalised rows, so it is
 * scale invariant per row and its gradient is tangent).  E_t = 0 means every
 * cosine is <= t: a *feasible* configuration.  A positive local minimum means
 * the basin holds no feasible point within reach of a descent method.
 *
 * Only pairs with cosine > t - margin can contribute, so they are kept in a
 * Verlet list; it is rebuilt whenever the largest displacement since the
 * last build could have let an unlisted pair cross t (two unit vectors moved
 * by d1, d2 change their inner product by at most d1 + d2 + d1 d2).
 *
 * usage: hingepol n N IN OUT [--t 0.5] [--iters 20000] [--margin 0.02]
 *                            [--mem 20] [--quiet] [--fixed K]
 *   --fixed K   keep the first K rows fixed (e.g. polish only reinserted points)
 * Prints one line: E, max cosine, iterations.  NEVER build with -ffast-math.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int n, N, nfixed = 0;
static size_t Nn;
static double T = 0.5, MARGIN = 0.02;
static int *pi_, *pj_;
static size_t np_, cap;
static double *Xb; /* normalised positions at the last list build */
static long n_build = 0;

static double wall(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }

static void normalised(const double *X, double *U) {
    for (int i = 0; i < N; i++) {
        const double *x = X + (size_t)i * n; double *u = U + (size_t)i * n, q = 0;
        for (int k = 0; k < n; k++) q += x[k] * x[k];
        q = 1.0 / sqrt(q);
        for (int k = 0; k < n; k++) u[k] = x[k] * q;
    }
}

static void build(const double *U) {
    size_t c = 0;
    double thr = T - MARGIN;
    for (int i = 0; i < N; i++) {
        const double *a = U + (size_t)i * n;
        for (int j = i + 1; j < N; j++) {
            if (i < nfixed && j < nfixed) continue; /* fixed-fixed pairs never move */
            const double *b = U + (size_t)j * n; double g = 0;
            for (int k = 0; k < n; k++) g += a[k] * b[k];
            if (g > thr) {
                if (c == cap) { cap = cap ? 2 * cap : 4096; pi_ = realloc(pi_, cap * sizeof(int)); pj_ = realloc(pj_, cap * sizeof(int)); }
                pi_[c] = i; pj_[c] = j; c++;
            }
        }
    }
    np_ = c;
    memcpy(Xb, U, sizeof(double) * Nn);
    n_build++;
}

static int list_stale(const double *U) {
    double m1 = 0, m2 = 0;
    for (int i = 0; i < N; i++) {
        const double *u = U + (size_t)i * n, *v = Xb + (size_t)i * n; double q = 0;
        for (int k = 0; k < n; k++) { double d = u[k] - v[k]; q += d * d; }
        q = sqrt(q);
        if (q > m1) { m2 = m1; m1 = q; } else if (q > m2) m2 = q;
    }
    return m1 + m2 + m1 * m2 >= MARGIN;
}

/* E and gradient w.r.t. raw X.  U is scratch for the normalised view. */
static double *Ubuf, *GU;
static double energy(const double *X, double *G, double *mx) {
    normalised(X, Ubuf);
    if (list_stale(Ubuf)) build(Ubuf);
    double E = 0, m = -2;
    memset(GU, 0, sizeof(double) * Nn);
    for (size_t p = 0; p < np_; p++) {
        int i = pi_[p], j = pj_[p];
        const double *a = Ubuf + (size_t)i * n, *b = Ubuf + (size_t)j * n;
        double g = 0;
        for (int k = 0; k < n; k++) g += a[k] * b[k];
        if (g > m) m = g;
        double h = g - T;
        if (h <= 0) continue;
        E += h * h;
        double c = 2 * h;
        double *ga = GU + (size_t)i * n, *gb = GU + (size_t)j * n;
        for (int k = 0; k < n; k++) { ga[k] += c * b[k]; gb[k] += c * a[k]; }
    }
    if (mx) *mx = m;
    if (G) {
        for (int i = 0; i < N; i++) {
            const double *x = X + (size_t)i * n, *u = Ubuf + (size_t)i * n;
            double *g = G + (size_t)i * n, *gu = GU + (size_t)i * n, q = 0, d = 0;
            if (i < nfixed) { memset(g, 0, sizeof(double) * n); continue; }
            for (int k = 0; k < n; k++) { q += x[k] * x[k]; d += gu[k] * u[k]; }
            double r = 1.0 / sqrt(q);
            for (int k = 0; k < n; k++) g[k] = (gu[k] - d * u[k]) * r;
        }
    }
    return E;
}

static double dotv(const double *a, const double *b) { double s = 0; for (size_t t = 0; t < Nn; t++) s += a[t] * b[t]; return s; }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: hingepol n N IN OUT [--t T] [--iters K] [--margin M] [--mem m] [--fixed K] [--quiet]\n"); return 1; }
    n = atoi(argv[1]); N = atoi(argv[2]); Nn = (size_t)N * n;
    const char *in = argv[3], *out = argv[4];
    long iters = 20000; int mem = 20, quiet = 0;
    for (int a = 5; a < argc; a++) {
        if (!strcmp(argv[a], "--t") && a + 1 < argc) T = atof(argv[++a]);
        else if (!strcmp(argv[a], "--iters") && a + 1 < argc) iters = atol(argv[++a]);
        else if (!strcmp(argv[a], "--margin") && a + 1 < argc) MARGIN = atof(argv[++a]);
        else if (!strcmp(argv[a], "--mem") && a + 1 < argc) mem = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--fixed") && a + 1 < argc) nfixed = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--quiet")) quiet = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 1; }
    }
    double *X = malloc(sizeof(double) * Nn);
    FILE *f = fopen(in, "r"); if (!f) { perror(in); return 1; }
    size_t c = 0; char *line = NULL; size_t ln = 0;
    while (getline(&line, &ln, f) > 0) {
        if (line[0] == '#') continue;
        char *p = line, *e;
        for (;;) { double v = strtod(p, &e); if (e == p) break; if (c < Nn) X[c] = v; c++; p = e; }
    }
    fclose(f); free(line);
    if (c != Nn) { fprintf(stderr, "%s: read %zu coordinates, want %zu\n", in, c, Nn); return 1; }
    for (size_t t = 0; t < Nn; t++) if (!isfinite(X[t])) { fprintf(stderr, "non-finite input\n"); return 1; }
    normalised(X, X);

    Ubuf = malloc(sizeof(double) * Nn); GU = malloc(sizeof(double) * Nn); Xb = malloc(sizeof(double) * Nn);
    double *G = malloc(sizeof(double) * Nn), *Gn = malloc(sizeof(double) * Nn), *D = malloc(sizeof(double) * Nn), *Xn = malloc(sizeof(double) * Nn);
    double *S = malloc(sizeof(double) * Nn * mem), *Y = malloc(sizeof(double) * Nn * mem), *rho = malloc(sizeof(double) * mem), *alp = malloc(sizeof(double) * mem);
    normalised(X, Ubuf); build(Ubuf);

    double t0 = wall(), mx;
    double E = energy(X, G, &mx);
    int k = 0, head = 0; long it;
    for (it = 0; it < iters && E > 0; it++) {
        /* two-loop recursion */
        memcpy(D, G, sizeof(double) * Nn);
        for (int q = 0; q < k; q++) {
            int idx = (head - 1 - q + mem) % mem;
            alp[idx] = rho[idx] * dotv(S + (size_t)idx * Nn, D);
            double *yy = Y + (size_t)idx * Nn;
            for (size_t t = 0; t < Nn; t++) D[t] -= alp[idx] * yy[t];
        }
        if (k > 0) {
            int last = (head - 1 + mem) % mem;
            double gam = dotv(S + (size_t)last * Nn, Y + (size_t)last * Nn) / dotv(Y + (size_t)last * Nn, Y + (size_t)last * Nn);
            for (size_t t = 0; t < Nn; t++) D[t] *= gam;
        } else {
            double gn = sqrt(dotv(G, G));
            double sc = gn > 0 ? 1e-3 / gn : 0;
            for (size_t t = 0; t < Nn; t++) D[t] *= sc;
        }
        for (int q = k - 1; q >= 0; q--) {
            int idx = (head - 1 - q + mem) % mem;
            double b = rho[idx] * dotv(Y + (size_t)idx * Nn, D);
            double *ss = S + (size_t)idx * Nn;
            for (size_t t = 0; t < Nn; t++) D[t] += (alp[idx] - b) * ss[t];
        }
        double slope = -dotv(G, D);
        if (!(slope < 0)) { k = 0; continue; } /* not a descent direction: reset */
        /* backtracking Armijo along -D */
        double step = 1.0, En = 0, mxn = 0;
        int ok = 0;
        for (int ls = 0; ls < 40; ls++) {
            for (size_t t = 0; t < Nn; t++) Xn[t] = X[t] - step * D[t];
            En = energy(Xn, Gn, &mxn);
            if (En <= E + 1e-4 * step * slope) { ok = 1; break; }
            step *= 0.5;
        }
        if (!ok) { if (k == 0) break; k = 0; continue; }
        int idx = head;
        double *ss = S + (size_t)idx * Nn, *yy = Y + (size_t)idx * Nn;
        for (size_t t = 0; t < Nn; t++) { ss[t] = Xn[t] - X[t]; yy[t] = Gn[t] - G[t]; }
        double sy = dotv(ss, yy);
        if (sy > 1e-300) { rho[idx] = 1.0 / sy; head = (head + 1) % mem; if (k < mem) k++; }
        memcpy(X, Xn, sizeof(double) * Nn); memcpy(G, Gn, sizeof(double) * Nn);
        E = En; mx = mxn;
        if (!quiet && it % 1000 == 0) fprintf(stderr, "it %ld E %.6e max %.15f pairs %zu\n", it, E, mx, np_);
        if (it % 200 == 199) { /* keep raw norms near 1; resets the memory, so only when needed */
            double dev = 0;
            for (int i = 0; i < N; i++) { double q = 0; for (int r = 0; r < n; r++) q += X[(size_t)i * n + r] * X[(size_t)i * n + r]; q = fabs(sqrt(q) - 1); if (q > dev) dev = q; }
            if (dev > 0.1) { normalised(X, X); k = 0; E = energy(X, G, &mx); }
        }
    }
    normalised(X, X);
    E = energy(X, NULL, &mx);
    /* report the true max over all pairs, not just listed ones */
    double m = -2;
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
        double g = 0; for (int q = 0; q < n; q++) g += X[(size_t)i * n + q] * X[(size_t)j * n + q];
        if (g > m) m = g;
    }
    printf("E=%.17g max=%.17g iters=%ld time=%.2f builds=%ld pairs=%zu\n", E, m, it, wall() - t0, n_build, np_);
    FILE *o = fopen(out, "w"); if (!o) { perror(out); return 1; }
    fprintf(o, "# n=%d N=%d max_inner=%.17g hinge_t=%.17g E=%.17g\n", n, N, m, T, E);
    for (int i = 0; i < N; i++) for (int q = 0; q < n; q++) fprintf(o, "%.17g%c", X[(size_t)i * n + q], q + 1 < n ? ' ' : '\n');
    fclose(o);
    return 0;
}
