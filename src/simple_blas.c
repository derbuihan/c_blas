// SPDX-License-Identifier: MIT
#include "simple_blas.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

static inline size_t row_major_index(int row, int col, int ld) {
    return (size_t)row * (size_t)ld + (size_t)col;
}

static inline size_t col_major_index(int row, int col, int ld) {
    return (size_t)row + (size_t)col * (size_t)ld;
}

static void zero_matrix(int M, int N, float *C, int ldc, bool row_major) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            size_t idx = row_major ? row_major_index(i, j, ldc)
                                   : col_major_index(i, j, ldc);
            C[idx] = 0.0f;
        }
    }
}

static void scale_matrix(int M, int N, float *C, int ldc, float beta, bool row_major) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            size_t idx = row_major ? row_major_index(i, j, ldc)
                                   : col_major_index(i, j, ldc);
            C[idx] *= beta;
        }
    }
}

static void reference_sgemm(bool row_major,
                            bool trans_a,
                            bool trans_b,
                            int M,
                            int N,
                            int K,
                            float alpha,
                            const float *A,
                            int lda,
                            const float *B,
                            int ldb,
                            float beta,
                            float *C,
                            int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                int a_row = trans_a ? p : i;
                int a_col = trans_a ? i : p;
                int b_row = trans_b ? j : p;
                int b_col = trans_b ? p : j;

                size_t a_idx = row_major ? row_major_index(a_row, a_col, lda)
                                         : col_major_index(a_row, a_col, lda);
                size_t b_idx = row_major ? row_major_index(b_row, b_col, ldb)
                                         : col_major_index(b_row, b_col, ldb);

                acc += A[a_idx] * B[b_idx];
            }

            size_t c_idx = row_major ? row_major_index(i, j, ldc)
                                     : col_major_index(i, j, ldc);
            float value = alpha * acc;
            if (beta == 1.0f) {
                value += C[c_idx];
            } else if (beta != 0.0f) {
                value += beta * C[c_idx];
            }
            C[c_idx] = value;
        }
    }
}

#if defined(__aarch64__)
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

void simple_blas_arm64_kernel_12x8(const float *A,
                                   const float *B,
                                   float *C,
                                   int lda,
                                   int ldb,
                                   int ldc,
                                   int K,
                                   float alpha,
                                   float beta);

// ASM packing functions
int simple_blas_arm64_pack_A_12x4(int K, const float *A, int lda, float *buffer);
void simple_blas_arm64_pack_B_8xK(int K, const float *B, int ldb, float *buffer);

#define SIMPLE_BLAS_ARM64_TILE_M 12
#define SIMPLE_BLAS_ARM64_TILE_N 8

// Cache blocking parameters (tuned for Apple Silicon / ARM64)
#define MC 192
#define KC 192
#define NC 384

static void pack_A(int M, int K, const float *A, int lda, float *buffer) {
    int i = 0;
    for (; i + SIMPLE_BLAS_ARM64_TILE_M <= M; i += SIMPLE_BLAS_ARM64_TILE_M) {
        int k_packed = simple_blas_arm64_pack_A_12x4(K, A + (size_t)i * lda, lda, buffer);
        buffer += 12 * k_packed;
        for (int p = k_packed; p < K; ++p) {
            const float *a_ptr = A + (size_t)i * lda + p;
            for (int k = 0; k < SIMPLE_BLAS_ARM64_TILE_M; ++k) {
                *buffer++ = a_ptr[k * lda];
            }
        }
    }
    if (i < M) {
        for (int p = 0; p < K; ++p) {
            const float *a_ptr = A + (size_t)i * lda + p;
            for (int k = 0; k < SIMPLE_BLAS_ARM64_TILE_M; ++k) {
                if (i + k < M) {
                    *buffer++ = a_ptr[k * lda];
                } else {
                    *buffer++ = 0.0f;
                }
            }
        }
    }
}

static void pack_B(int K, int N, const float *B, int ldb, float *buffer) {
    int j = 0;
    for (; j + SIMPLE_BLAS_ARM64_TILE_N <= N; j += SIMPLE_BLAS_ARM64_TILE_N) {
        simple_blas_arm64_pack_B_8xK(K, B + j, ldb, buffer);
        buffer += K * 8;
    }
    if (j < N) {
        for (int p = 0; p < K; ++p) {
            const float *b_ptr = B + (size_t)p * ldb + j;
            for (int k = 0; k < SIMPLE_BLAS_ARM64_TILE_N; ++k) {
                if (j + k < N) {
                    *buffer++ = b_ptr[k];
                } else {
                    *buffer++ = 0.0f;
                }
            }
        }
    }
}

static void scalar_tail_block(int row_offset,
                              int col_offset,
                              int rows,
                              int cols,
                              int K,
                              float alpha,
                              const float *A,
                              int lda,
                              const float *B,
                              int ldb,
                              float beta,
                              float *C,
                              int ldc) {
    for (int ii = 0; ii < rows; ++ii) {
        const float *a_row = A + (size_t)(row_offset + ii) * lda;
        float *c_row = C + (size_t)(row_offset + ii) * ldc + col_offset;
        for (int jj = 0; jj < cols; ++jj) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                size_t b_idx = (size_t)p * ldb + (col_offset + jj);
                acc += a_row[p] * B[b_idx];
            }


            float value = alpha * acc;
            if (beta == 1.0f) {
                value += c_row[jj];
            } else if (beta != 0.0f) {
                value += beta * c_row[jj];
            }
            c_row[jj] = value;
        }
    }
}

static void arm64_row_major_sgemm(int M,
                                  int N,
                                  int K,
                                  float alpha,
                                  const float *A,
                                  int lda,
                                  const float *B,
                                  int ldb,
                                  float beta,
                                  float *C,
                                  int ldc) {
    size_t size_A = (size_t)MC * (size_t)KC * sizeof(float);
    size_t size_B = (size_t)KC * (size_t)NC * sizeof(float);
    float *packed_A = malloc(size_A);
    float *packed_B = malloc(size_B);
    if (!packed_A || !packed_B) {
        free(packed_A); free(packed_B);
        reference_sgemm(true, false, false, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    for (int j = 0; j < N; j += NC) {
        int jb = (N - j > NC) ? NC : N - j;
        for (int p = 0; p < K; p += KC) {
            int pb = (K - p > KC) ? KC : K - p;
            pack_B(pb, jb, B + (size_t)p * ldb + j, ldb, packed_B);
            for (int i = 0; i < M; i += MC) {
                int ib = (M - i > MC) ? MC : M - i;
                pack_A(ib, pb, A + (size_t)i * lda + p, lda, packed_A);
                for (int jr = 0; jr < jb; jr += SIMPLE_BLAS_ARM64_TILE_N) {
                    for (int ir = 0; ir < ib; ir += SIMPLE_BLAS_ARM64_TILE_M) {
                        float *ptr_c = C + (size_t)(i + ir) * ldc + (j + jr);
                        float current_beta = (p == 0) ? beta : 1.0f;
                        if (ir + SIMPLE_BLAS_ARM64_TILE_M <= ib && jr + SIMPLE_BLAS_ARM64_TILE_N <= jb) {
                            simple_blas_arm64_kernel_12x8(packed_A + (size_t)ir * pb, packed_B + (size_t)jr * pb, ptr_c, 0, 8, ldc, pb, alpha, current_beta);
                        } else {
                            int me = (ib - ir > 12) ? 12 : ib - ir;
                            int ne = (jb - jr > 8) ? 8 : jb - jr;
                            scalar_tail_block(i + ir, j + jr, me, ne, pb, alpha, A + (size_t)p, lda, B + (size_t)p * ldb, ldb, current_beta, C, ldc);
                        }
                    }
                }
            }
        }
    }
    free(packed_A); free(packed_B);
}

#define MT_THRESHOLD_M 256
#define MAX_THREADS 16

static int simple_blas_thread_cap(void) {
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs < 1) nprocs = 1;
    if (nprocs > MAX_THREADS) nprocs = MAX_THREADS;
    const char *env = getenv("SIMPLE_SGEMM_THREADS");
    if (env && *env) {
        char *end = NULL;
        long override = strtol(env, &end, 10);
        if (end != env && override > 0) {
            if (override > MAX_THREADS) override = MAX_THREADS;
            nprocs = override;
        }
    }
    return (int)nprocs;
}

typedef struct {
    int M, N, K;
    float alpha, beta;
    const float *A, *B;
    int lda, ldb, ldc;
    float *C;
} ThreadArgs;

static void *gemm_thread_worker(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;
    arm64_row_major_sgemm(a->M, a->N, a->K, a->alpha, a->A, a->lda, a->B, a->ldb, a->beta, a->C, a->ldc);
    return NULL;
}

static bool arm64_try_fast_path(bool rm, bool ta, bool tb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    if (!rm || ta || tb) return false;
    int nt = simple_blas_thread_cap();
    if (M < MT_THRESHOLD_M || nt <= 1) {
        arm64_row_major_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return true;
    }
    pthread_t t[MAX_THREADS]; ThreadArgs args[MAX_THREADS];
    int rpt = M / nt, rem = M % nt, cur = 0;
    for (int i = 0; i < nt; i++) {
        int rows = rpt + (i < rem ? 1 : 0);
        args[i] = (ThreadArgs){rows, N, K, alpha, beta, A + (size_t)cur * lda, B, lda, ldb, ldc, C + (size_t)cur * ldc};
        pthread_create(&t[i], NULL, gemm_thread_worker, &args[i]);
        cur += rows;
    }
    for (int i = 0; i < nt; i++) pthread_join(t[i], NULL);
    return true;
}
#endif

void simple_cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    bool rm = (Order == CblasRowMajor), ta = (TransA == CblasTrans || TransA == CblasConjTrans), tb = (TransB == CblasTrans || TransB == CblasConjTrans);
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (alpha == 0.0f) { if (beta == 0.0f) zero_matrix(M, N, C, ldc, rm); else if (beta != 1.0f) scale_matrix(M, N, C, ldc, beta, rm); return; }
#if defined(__aarch64__)
    if (arm64_try_fast_path(rm, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)) return;
#endif
    reference_sgemm(rm, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#ifndef SIMPLE_BLAS_NO_ALIAS
void cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    simple_cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif
