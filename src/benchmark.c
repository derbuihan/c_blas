// SPDX-License-Identifier: MIT
#define _GNU_SOURCE
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

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void run_sgemm_batch(sgemm_fn fn, int n, int iterations, const float *A, const float *B, float *C, float beta) {
    for (int i = 0; i < iterations; ++i) {
        fn(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, beta, C, n);
    }
}

static void print_backend_info(const char *label, sgemm_fn fn) {
    if (!fn) return;
    Dl_info info;
    if (dladdr((void *)fn, &info) && info.dli_fname) {
        printf("  [%s] symbol from: %s\n", label, info.dli_fname);
    }
}

static void benchmark_backend(const char *label,
                              sgemm_fn fn,
                              float *C_work,
                              const float *C_seed,
                              size_t elems,
                              const float *A,
                              const float *B,
                              int n) {
    if (!fn) return;

    const double target_bench_time = 0.2; 
    const int outer_repeats = 5;         
    const size_t bytes = elems * sizeof(float);
    const float beta = 1.0f; // Force reading C
    
    // Warm up & Estimate iterations
    run_sgemm_batch(fn, n, 1, A, B, C_work, beta); 
    
    double start_est = monotonic_seconds();
    run_sgemm_batch(fn, n, 1, A, B, C_work, beta);
    double single_run = monotonic_seconds() - start_est;
    
    int iterations = 1;
    if (single_run < target_bench_time) {
        iterations = (int)(target_bench_time / (single_run + 1e-9));
        if (iterations < 1) iterations = 1;
        if (iterations > 1000000) iterations = 1000000;
    }

    double results[outer_repeats];
    double total_elapsed_sum = 0;
    
    // To prevent compiler from optimizing away the whole loop
    volatile float sink = 0.0f;

    for (int r = 0; r < outer_repeats; ++r) {
        memcpy(C_work, C_seed, bytes);
        
        double start = monotonic_seconds();
        run_sgemm_batch(fn, n, iterations, A, B, C_work, beta);
        double elapsed = monotonic_seconds() - start;
        
        results[r] = elapsed / iterations;
        total_elapsed_sum += elapsed;
        
        // Force a read from the result matrix
        sink += C_work[rand() % elems];
    }

    qsort(results, outer_repeats, sizeof(double), compare_doubles);
    
    double median_time = results[outer_repeats / 2];
    double flops_per_gemm = 2.0 * pow(n, 3);
    double median_gflops = (flops_per_gemm / median_time) / 1e9;

    printf("  %-12s | Median: %10.6f s (%8.2f GFLOP/s) | Total: %4.2fs | Iters: %-7d\n",
           label, median_time, median_gflops, total_elapsed_sum, iterations);
}

static float *allocate_matrix(size_t elements) {
    float *ptr = NULL;
    if (posix_memalign((void **)&ptr, 64, elements * sizeof(float)) != 0) return NULL;
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
    const char *path = backend->env_var ? getenv(backend->env_var) : NULL;
    if (!path || path[0] == '\0') path = backend->default_path;
    
    backend->handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!backend->handle) {
        backend->fn = NULL;
        return -1;
    }
    backend->fn = (sgemm_fn)dlsym(backend->handle, "cblas_sgemm");
    if (!backend->fn) {
        dlclose(backend->handle);
        backend->handle = NULL;
        return -1;
    }
    return 0;
}

int main(void) {
    srand(0);
    printf("SGEMM Benchmark (Target: 0.2s/sample, Beta=1.0, Median of 5)\n");

    dyn_backend openblas = {"OpenBLAS", "OPENBLAS_DYLIB", "/opt/homebrew/opt/openblas/lib/libopenblas.dylib", NULL, NULL};
    dyn_backend accelerate = {"Accelerate", "ACCELERATE_DYLIB", "/System/Library/Frameworks/Accelerate.framework/Accelerate", NULL, NULL};

    load_backend(&openblas);
    load_backend(&accelerate);

    print_backend_info("OpenBLAS", openblas.fn);
    print_backend_info("Accelerate", accelerate.fn);

    static const int sizes[] = {128, 256, 512, 1024, 2048};

    for (size_t idx = 0; idx < sizeof(sizes) / sizeof(sizes[0]); ++idx) {
        const int n = sizes[idx];
        const size_t elems = (size_t)n * n;

        float *A = allocate_matrix(elems);
        float *B = allocate_matrix(elems);
        float *C_seed = allocate_matrix(elems);
        float *C_simple = allocate_matrix(elems);
        float *C_ref = allocate_matrix(elems);

        if (!A || !B || !C_seed || !C_simple || !C_ref) return EXIT_FAILURE;

        fill_random(A, elems);
        fill_random(B, elems);
        fill_random(C_seed, elems);

        printf("\nSize %4d x %4d:\n", n, n);

        sgemm_fn ref_fn = accelerate.fn ? accelerate.fn : (openblas.fn ? openblas.fn : NULL);
        if (ref_fn) {
            ref_fn(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C_ref, n);
        }

        benchmark_backend("simple", simple_cblas_sgemm, C_simple, C_seed, elems, A, B, n);
        benchmark_backend("OpenBLAS", openblas.fn, C_simple, C_seed, elems, A, B, n);
        benchmark_backend("Accelerate", accelerate.fn, C_simple, C_seed, elems, A, B, n);

        if (ref_fn) {
            simple_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C_simple, n);
            float diff = max_abs_diff(C_simple, C_ref, elems);
            printf("  Verification max |Δ| = %.6e\n", diff);
        }

        free(A); free(B); free(C_seed); free(C_simple); free(C_ref);
    }

    if (openblas.handle) dlclose(openblas.handle);
    if (accelerate.handle) dlclose(accelerate.handle);

    return EXIT_SUCCESS;
}
