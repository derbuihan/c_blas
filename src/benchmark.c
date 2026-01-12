// SPDX-License-Identifier: MIT
#include "simple_blas.h"

#include <assert.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*sgemm_fn)(enum CBLAS_ORDER,
                         enum CBLAS_TRANSPOSE,
                         enum CBLAS_TRANSPOSE,
                         int,
                         int,
                         int,
                         float,
                         const float *,
                         int,
                         const float *,
                         int,
                         float,
                         float *,
                         int);

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_random(float *data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
}

static float max_abs_diff(const float *a, const float *b, size_t count) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

static double benchmark_impl(sgemm_fn fn,
                             float *C,
                             const float *A,
                             const float *B,
                             int size,
                             int iterations) {
    const int M = size;
    const int N = size;
    const int K = size;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    double best = 1e9;
    for (int it = 0; it < iterations; ++it) {
        double start = monotonic_seconds();
        fn(CblasRowMajor,
           CblasNoTrans,
           CblasNoTrans,
           M,
           N,
           K,
           alpha,
           A,
           K,
           B,
           N,
           beta,
           C,
           N);
        double elapsed = monotonic_seconds() - start;
        if (elapsed < best) {
            best = elapsed;
        }
    }

    return best;
}

static void benchmark_backend(const char *label,
                              sgemm_fn fn,
                              float *C_work,
                              const float *C_seed,
                              size_t elems,
                              const float *A,
                              const float *B,
                              int size,
                              int iterations,
                              int repeats) {
    if (!fn) {
        return;
    }
    const size_t bytes = elems * sizeof(float);
    double best = 1e9;
    double total = 0.0;

    for (int r = 0; r < repeats; ++r) {
        memcpy(C_work, C_seed, bytes);
        double t = benchmark_impl(fn, C_work, A, B, size, iterations);
        if (t < best) {
            best = t;
        }
        total += t;
    }

    double avg = total / repeats;
    double flops = 2.0 * (double)size * (double)size * (double)size;
    double best_gflops = flops / best / 1e9;
    double avg_gflops = flops / avg / 1e9;
    printf("  %-12s best %8.4f s  %8.3f GFLOP/s  avg %8.4f s  %8.3f GFLOP/s\n",
           label,
           best,
           best_gflops,
           avg,
           avg_gflops);
}

static float *allocate_matrix(size_t elements) {
    float *ptr = NULL;
    int rc = posix_memalign((void **)&ptr, 64, elements * sizeof(float));
    if (rc != 0) {
        return NULL;
    }
    return ptr;
}

typedef struct {
    const char *label;
    const char *env_var;
    const char *default_path;
    void *handle;
    sgemm_fn fn;
} dyn_backend;

static int load_backend(dyn_backend *backend) {
    const char *path = NULL;
    if (backend->env_var) {
        path = getenv(backend->env_var);
    }
    if (!path || path[0] == '\0') {
        path = backend->default_path;
    }
    backend->handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!backend->handle) {
        fprintf(stderr,
                "Skipping %s: failed to load %s (%s)\n",
                backend->label,
                path,
                dlerror());
        backend->fn = NULL;
        return -1;
    }
    backend->fn = (sgemm_fn)dlsym(backend->handle, "cblas_sgemm");
    if (!backend->fn) {
        fprintf(stderr,
                "Skipping %s: unable to find cblas_sgemm in %s\n",
                backend->label,
                path);
        dlclose(backend->handle);
        backend->handle = NULL;
        return -1;
    }
    return 0;
}

static void unload_backend(dyn_backend *backend) {
    if (backend->handle) {
        dlclose(backend->handle);
        backend->handle = NULL;
        backend->fn = NULL;
    }
}

int main(void) {
    srand(0);

    const int sizes[] = {512, 1024, 1536, 2048};
    const int iterations = 3;
    const int repeats = 5;

    dyn_backend openblas = {
        .label = "OpenBLAS",
        .env_var = "OPENBLAS_DYLIB",
        .default_path = "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
        .handle = NULL,
        .fn = NULL};

    dyn_backend accelerate = {
        .label = "Accelerate",
        .env_var = "ACCELERATE_DYLIB",
        .default_path = "/System/Library/Frameworks/Accelerate.framework/Accelerate",
        .handle = NULL,
        .fn = NULL};

    load_backend(&openblas);
    load_backend(&accelerate);

    for (size_t idx = 0; idx < sizeof(sizes) / sizeof(sizes[0]); ++idx) {
        const int n = sizes[idx];
        const size_t elems = (size_t)n * (size_t)n;

        float *A = allocate_matrix(elems);
        float *B = allocate_matrix(elems);
        float *C_seed = allocate_matrix(elems);
        float *C_simple = allocate_matrix(elems);
        float *C_blas = allocate_matrix(elems);

        if (!A || !B || !C_seed || !C_simple || !C_blas) {
            fprintf(stderr, "Failed to allocate matrices for size %d\n", n);
            return EXIT_FAILURE;
        }

        fill_random(A, elems);
        fill_random(B, elems);
        fill_random(C_seed, elems);
        memcpy(C_simple, C_seed, elems * sizeof(float));
        memcpy(C_blas, C_seed, elems * sizeof(float));

        printf("Size %4d x %4d:\n", n, n);
        benchmark_backend("simple",
                          simple_cblas_sgemm,
                          C_simple,
                          C_seed,
                          elems,
                          A,
                          B,
                          n,
                          iterations,
                          repeats);
        benchmark_backend(openblas.label,
                          openblas.fn,
                          C_blas,
                          C_seed,
                          elems,
                          A,
                          B,
                          n,
                          iterations,
                          repeats);
        benchmark_backend(accelerate.label,
                          accelerate.fn,
                          C_blas,
                          C_seed,
                          elems,
                          A,
                          B,
                          n,
                          iterations,
                          repeats);

        float diff = max_abs_diff(C_simple, C_blas, elems);
        printf("  max |Δ| = %.6f\n\n", diff);

        free(A);
        free(B);
        free(C_seed);
        free(C_simple);
        free(C_blas);
    }

    unload_backend(&openblas);
    unload_backend(&accelerate);

    return EXIT_SUCCESS;
}
