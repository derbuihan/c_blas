// SPDX-License-Identifier: MIT
#include "simple_blas.h"

#include <stdbool.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

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

static inline void neon_kernel_4x4(int i,
                                   int j,
                                   int K,
                                   float alpha,
                                   const float *A,
                                   int lda,
                                   const float *B,
                                   int ldb,
                                   float beta,
                                   float *C,
                                   int ldc) {
    const float *a0 = A + (size_t)(i + 0) * lda;
    const float *a1 = A + (size_t)(i + 1) * lda;
    const float *a2 = A + (size_t)(i + 2) * lda;
    const float *a3 = A + (size_t)(i + 3) * lda;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    const float *b_base = B + j;
    for (int p = 0; p < K; ++p) {
        const float *b_row = b_base + (size_t)p * ldb;
        float32x4_t vb = vld1q_f32(b_row);
        float a0_val = a0[p];
        float a1_val = a1[p];
        float a2_val = a2[p];
        float a3_val = a3[p];
        acc0 = vfmaq_n_f32(acc0, vb, a0_val);
        acc1 = vfmaq_n_f32(acc1, vb, a1_val);
        acc2 = vfmaq_n_f32(acc2, vb, a2_val);
        acc3 = vfmaq_n_f32(acc3, vb, a3_val);
    }

    float32x4_t alpha_v = vdupq_n_f32(alpha);
    acc0 = vmulq_f32(acc0, alpha_v);
    acc1 = vmulq_f32(acc1, alpha_v);
    acc2 = vmulq_f32(acc2, alpha_v);
    acc3 = vmulq_f32(acc3, alpha_v);

    float *c0 = C + (size_t)(i + 0) * ldc + j;
    float *c1 = C + (size_t)(i + 1) * ldc + j;
    float *c2 = C + (size_t)(i + 2) * ldc + j;
    float *c3 = C + (size_t)(i + 3) * ldc + j;

    if (beta == 0.0f) {
        vst1q_f32(c0, acc0);
        vst1q_f32(c1, acc1);
        vst1q_f32(c2, acc2);
        vst1q_f32(c3, acc3);
    } else {
        float32x4_t beta_v = vdupq_n_f32(beta);
        float32x4_t c_vec0 = vld1q_f32(c0);
        float32x4_t c_vec1 = vld1q_f32(c1);
        float32x4_t c_vec2 = vld1q_f32(c2);
        float32x4_t c_vec3 = vld1q_f32(c3);
        if (beta == 1.0f) {
            acc0 = vaddq_f32(acc0, c_vec0);
            acc1 = vaddq_f32(acc1, c_vec1);
            acc2 = vaddq_f32(acc2, c_vec2);
            acc3 = vaddq_f32(acc3, c_vec3);
        } else {
            acc0 = vmlaq_f32(acc0, c_vec0, beta_v);
            acc1 = vmlaq_f32(acc1, c_vec1, beta_v);
            acc2 = vmlaq_f32(acc2, c_vec2, beta_v);
            acc3 = vmlaq_f32(acc3, c_vec3, beta_v);
        }
        vst1q_f32(c0, acc0);
        vst1q_f32(c1, acc1);
        vst1q_f32(c2, acc2);
        vst1q_f32(c3, acc3);
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
    for (int i = 0; i < M; i += SIMPLE_BLAS_ARM64_TILE) {
        int rows = (M - i >= SIMPLE_BLAS_ARM64_TILE) ? SIMPLE_BLAS_ARM64_TILE : (M - i);
        for (int j = 0; j < N; j += SIMPLE_BLAS_ARM64_TILE) {
            int cols = (N - j >= SIMPLE_BLAS_ARM64_TILE) ? SIMPLE_BLAS_ARM64_TILE : (N - j);
            if (rows == SIMPLE_BLAS_ARM64_TILE && cols == SIMPLE_BLAS_ARM64_TILE) {
                neon_kernel_4x4(i, j, K, alpha, A, lda, B, ldb, beta, C, ldc);
            } else {
                scalar_tail_block(i, j, rows, cols, K, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        }
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
