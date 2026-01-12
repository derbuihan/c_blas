// SPDX-License-Identifier: MIT
#include "../src/simple_blas.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#define EPSILON 1e-4f

// Naive reference implementation for verification
static void naive_sgemm(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                acc += A[i * lda + p] * B[p * ldb + j]; // Assumes RowMajor, NoTrans
            }
            if (beta == 0.0f) {
                C[i * ldc + j] = alpha * acc;
            } else {
                C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
            }
        }
    }
}

static void fill_random(float *data, int count) {
    for (int i = 0; i < count; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
}

static bool check_matrices(int M, int N, const float *C_ref, const float *C_test, int ldc) {
    float max_diff = 0.0f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = fabsf(C_ref[i * ldc + j] - C_test[i * ldc + j]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > EPSILON) {
                fprintf(stderr, "Mismatch at (%d, %d): ref=%f, test=%f, diff=%f\n", i, j, C_ref[i * ldc + j], C_test[i * ldc + j], diff);
                return false;
            }
        }
    }
    return true;
}

static void run_test(int M, int N, int K, float alpha, float beta, const char *name) {
    printf("Testing %s (M=%d, N=%d, K=%d, alpha=%.1f, beta=%.1f)... ", name, M, N, K, alpha, beta); 

    int lda = K;
    int ldb = N;
    int ldc = N;

    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C_ref = malloc(M * N * sizeof(float));
    float *C_test = malloc(M * N * sizeof(float));

    fill_random(A, M * K);
    fill_random(B, K * N);
    
    // Fill C with random data to test beta accumulation
    fill_random(C_ref, M * N);
    for (int i = 0; i < M * N; ++i) C_test[i] = C_ref[i];

    // Run reference
    naive_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C_ref, ldc);

    // Run simple_blas
    simple_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C_test, ldc);

    if (check_matrices(M, N, C_ref, C_test, ldc)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        exit(EXIT_FAILURE);
    }

    free(A);
    free(B);
    free(C_ref);
    free(C_test);
}

int main(void) {
    // Basic tests
    run_test(1, 1, 1, 1.0f, 0.0f, "1x1x1");
    run_test(4, 4, 4, 1.0f, 0.0f, "4x4x4");
    
    // Kernel tile specific tests (Tile is 12x8)
    run_test(12, 8, 12, 1.0f, 0.0f, "Tile Exact (12x8)");
    run_test(12, 8, 128, 1.0f, 0.0f, "Tile Exact Long K");
    
    // Edge cases (around tile size)
    run_test(13, 9, 13, 1.0f, 0.0f, "Tile + 1");
    run_test(11, 7, 11, 1.0f, 0.0f, "Tile - 1");
    
    // Alpha / Beta tests
    run_test(16, 16, 16, 2.5f, 0.0f, "Alpha != 1, Beta = 0");
    run_test(16, 16, 16, 1.0f, 1.0f, "Alpha = 1, Beta = 1");
    run_test(16, 16, 16, 1.0f, 0.5f, "Alpha = 1, Beta = 0.5");
    run_test(16, 16, 16, 0.0f, 1.0f, "Alpha = 0 (No op A*B)");
    
    // Larger random tests
    run_test(64, 64, 64, 1.0f, 0.0f, "64x64x64");
    run_test(128, 128, 128, 1.0f, 0.0f, "128x128x128");
    run_test(255, 255, 255, 1.0f, 0.0f, "Odd Size 255");
    
    // K Unrolling boundary checks (Unroll factor 4)
    run_test(12, 8, 3, 1.0f, 0.0f, "K < 4");
    run_test(12, 8, 4, 1.0f, 0.0f, "K = 4");
    run_test(12, 8, 5, 1.0f, 0.0f, "K = 5");

    printf("All tests passed!\n");
    return 0;
}
