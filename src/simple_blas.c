// SPDX-License-Identifier: MIT
#include "simple_blas.h"

#include <stdbool.h>

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
void simple_blas_arm64_kernel_4x4(const float *A,
                                  const float *B,
                                  float *C,
                                  int lda,
                                  int ldb,
                                  int ldc,
                                  int K,
                                  float alpha,
                                  float beta);
#define SIMPLE_BLAS_ARM64_TILE 4

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
    const int tile = SIMPLE_BLAS_ARM64_TILE;
    int i = 0;
    for (; i + tile <= M; i += tile) {
        int j = 0;
        for (; j + tile <= N; j += tile) {
            const float *a_block = A + (size_t)i * lda;
            const float *b_block = B + j;
            float *c_block = C + (size_t)i * ldc + j;
            simple_blas_arm64_kernel_4x4(a_block,
                                         b_block,
                                         c_block,
                                         lda,
                                         ldb,
                                         ldc,
                                         K,
                                         alpha,
                                         beta);
        }
        if (j < N) {
            scalar_tail_block(i,
                              j,
                              tile,
                              N - j,
                              K,
                              alpha,
                              A,
                              lda,
                              B,
                              ldb,
                              beta,
                              C,
                              ldc);
        }
    }
    if (i < M) {
        scalar_tail_block(i,
                          0,
                          M - i,
                          N,
                          K,
                          alpha,
                          A,
                          lda,
                          B,
                          ldb,
                          beta,
                          C,
                          ldc);
    }
}

static bool arm64_try_fast_path(bool row_major,
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
    if (!row_major || trans_a || trans_b) {
        return false;
    }
    arm64_row_major_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return true;
}
#endif

void simple_cblas_sgemm(enum CBLAS_ORDER Order,
                        enum CBLAS_TRANSPOSE TransA,
                        enum CBLAS_TRANSPOSE TransB,
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
    const bool row_major = (Order == CblasRowMajor);
    const bool trans_a = (TransA == CblasTrans || TransA == CblasConjTrans);
    const bool trans_b = (TransB == CblasTrans || TransB == CblasConjTrans);

    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    if (alpha == 0.0f) {
        if (beta == 0.0f) {
            zero_matrix(M, N, C, ldc, row_major);
        } else if (beta != 1.0f) {
            scale_matrix(M, N, C, ldc, beta, row_major);
        }
        return;
    }

#if defined(__aarch64__)
    if (arm64_try_fast_path(row_major,
                            trans_a,
                            trans_b,
                            M,
                            N,
                            K,
                            alpha,
                            A,
                            lda,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc)) {
        return;
    }
#endif

    reference_sgemm(row_major,
                    trans_a,
                    trans_b,
                    M,
                    N,
                    K,
                    alpha,
                    A,
                    lda,
                    B,
                    ldb,
                    beta,
                    C,
                    ldc);
}

#ifndef SIMPLE_BLAS_NO_ALIAS
void cblas_sgemm(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB,
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
    simple_cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif
