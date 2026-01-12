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
                         int, int, int,
                         float, const float *, int,
                         const float *, int,
                         float, float *, int);

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
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

// 修正1: 各イテレーションでCを初期化するオプション
static void run_sgemm_batch(sgemm_fn fn, int n, int iterations,
                            const float *A, const float *B,
                            float *C, const float *C_init, size_t elems,
                            float beta, int reset_c_each_iter) {
    for (int i = 0; i < iterations; ++i) {
        if (reset_c_each_iter && C_init) {
            memcpy(C, C_init, elems * sizeof(float));
        }
        fn(CblasRowMajor, CblasNoTrans, CblasNoTrans,
           n, n, n, 1.0f, A, n, B, n, beta, C, n);
    }
}

static void print_backend_info(const char *label, sgemm_fn fn) {
    if (!fn) return;
    Dl_info info;
    if (dladdr((void *)fn, &info) && info.dli_fname) {
        printf("  [%s] symbol from: %s\n", label, info.dli_fname);
    }
}

// 修正2: 入力データも各バックエンドで独立に準備
static void benchmark_backend(const char *label,
                              sgemm_fn fn,
                              int n,
                              unsigned int seed) {
    if (!fn) return;

    const double target_bench_time = 0.2;
    const int outer_repeats = 5;
    const size_t elems = (size_t)n * n;
    const size_t bytes = elems * sizeof(float);
    const float beta = 0.0f;  // 修正3: beta=0でより公平な比較

    // 各バックエンド用に独立したメモリを確保
    float *A = NULL, *B = NULL, *C = NULL, *C_init = NULL;
    posix_memalign((void **)&A, 64, bytes);
    posix_memalign((void **)&B, 64, bytes);
    posix_memalign((void **)&C, 64, bytes);
    posix_memalign((void **)&C_init, 64, bytes);
    
    if (!A || !B || !C || !C_init) {
        printf("  %-12s | Memory allocation failed\n", label);
        free(A); free(B); free(C); free(C_init);
        return;
    }

    // 同じシードで初期化（再現性のため）
    srand(seed);
    fill_random(A, elems);
    fill_random(B, elems);
    fill_random(C_init, elems);

    // ウォームアップ（結果は捨てる）
    memcpy(C, C_init, bytes);
    fn(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       n, n, n, 1.0f, A, n, B, n, beta, C, n);
    
    // 修正4: ウォームアップ後にCをリセットしてから時間推定
    memcpy(C, C_init, bytes);
    double start_est = monotonic_seconds();
    fn(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       n, n, n, 1.0f, A, n, B, n, beta, C, n);
    double single_run = monotonic_seconds() - start_est;

    int iterations = (single_run < target_bench_time)
                   ? (int)(target_bench_time / (single_run + 1e-9))
                   : 1;
    if (iterations < 1) iterations = 1;
    if (iterations > 100000) iterations = 100000;

    double results[outer_repeats];
    double total_elapsed_sum = 0;
    volatile float sink = 0.0f;

    for (int r = 0; r < outer_repeats; ++r) {
        memcpy(C, C_init, bytes);

        double start = monotonic_seconds();
        // 修正5: beta=0なので各iterでCリセット不要（結果が上書きされる）
        run_sgemm_batch(fn, n, iterations, A, B, C, C_init, elems, beta, 0);
        double elapsed = monotonic_seconds() - start;

        results[r] = elapsed / iterations;
        total_elapsed_sum += elapsed;
        sink += C[rand() % elems];
    }

    (void)sink;
    qsort(results, outer_repeats, sizeof(double), compare_doubles);

    double median_time = results[outer_repeats / 2];
    // 修正6: FLOPの正確な計算 (beta=0の場合)
    double flops_per_gemm = 2.0 * (double)n * (double)n * (double)n;
    double median_gflops = (flops_per_gemm / median_time) / 1e9;

    printf("  %-12s | Median: %10.6f s (%8.2f GFLOP/s) | Total: %4.2fs | Iters: %-7d\n",
           label, median_time, median_gflops, total_elapsed_sum, iterations);

    free(A); free(B); free(C); free(C_init);
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
    if (!backend->handle) { backend->fn = NULL; return -1; }
    backend->fn = (sgemm_fn)dlsym(backend->handle, "cblas_sgemm");
    if (!backend->fn) { dlclose(backend->handle); backend->handle = NULL; return -1; }
    return 0;
}

int main(void) {
    const unsigned int base_seed = 42;
    printf("SGEMM Benchmark (Target: 0.2s/sample, Beta=0.0, Median of 5)\n");

    dyn_backend openblas = {"OpenBLAS", "OPENBLAS_DYLIB",
                            "/opt/homebrew/opt/openblas/lib/libopenblas.dylib", NULL, NULL};
    dyn_backend accelerate = {"Accelerate", "ACCELERATE_DYLIB",
                              "/System/Library/Frameworks/Accelerate.framework/Accelerate", NULL, NULL};

    load_backend(&openblas);
    load_backend(&accelerate);

    print_backend_info("OpenBLAS", openblas.fn);
    print_backend_info("Accelerate", accelerate.fn);

    static const int sizes[] = {128, 256, 512, 1024, 2048};

    for (size_t idx = 0; idx < sizeof(sizes) / sizeof(sizes[0]); ++idx) {
        const int n = sizes[idx];
        printf("\nSize %4d x %4d:\n", n, n);

        // 修正7: 各バックエンドに同じシードを渡して公平に
        benchmark_backend("simple", simple_cblas_sgemm, n, base_seed);
        benchmark_backend("OpenBLAS", openblas.fn, n, base_seed);
        benchmark_backend("Accelerate", accelerate.fn, n, base_seed);

        // 検証用（別途実行）
        const size_t elems = (size_t)n * n;
        float *A = allocate_matrix(elems);
        float *B = allocate_matrix(elems);
        float *C_simple = allocate_matrix(elems);
        float *C_ref = allocate_matrix(elems);

        if (A && B && C_simple && C_ref) {
            srand(base_seed);
            fill_random(A, elems);
            fill_random(B, elems);
            memset(C_simple, 0, elems * sizeof(float));
            memset(C_ref, 0, elems * sizeof(float));

            sgemm_fn ref_fn = accelerate.fn ? accelerate.fn
                            : (openblas.fn ? openblas.fn : NULL);
            if (ref_fn) {
                simple_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                   n, n, n, 1.0f, A, n, B, n, 0.0f, C_simple, n);
                ref_fn(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       n, n, n, 1.0f, A, n, B, n, 0.0f, C_ref, n);
                float diff = max_abs_diff(C_simple, C_ref, elems);
                printf("  Verification max |Δ| = %.6e\n", diff);
            }
        }
        free(A); free(B); free(C_simple); free(C_ref);
    }

    if (openblas.handle) dlclose(openblas.handle);
    if (accelerate.handle) dlclose(accelerate.handle);

    return EXIT_SUCCESS;
}
