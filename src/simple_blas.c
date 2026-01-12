// SPDX-License-Identifier: MIT
#include "simple_blas.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && !defined(SIMPLE_BLAS_DISABLE_ARM64)
#include <arm_neon.h>
#define SIMPLE_BLAS_ARM64 1
#else
#define SIMPLE_BLAS_ARM64 0
#endif

static inline size_t row_major_index(int row, int col, int ld) {
    return (size_t)row * (size_t)ld + (size_t)col;
}

static inline size_t col_major_index(int row, int col, int ld) {
    return (size_t)row + (size_t)col * (size_t)ld;
}

static inline int simple_min_int(int a, int b) {
    return (a < b) ? a : b;
}

#if SIMPLE_BLAS_ARM64
#define SIMPLE_MC 128
#define SIMPLE_NC 128
#define SIMPLE_KC 128
#define SIMPLE_MR 4
#define SIMPLE_NR 4

static float *simple_aligned_alloc(size_t count) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, count * sizeof(float)) != 0) {
        return NULL;
    }
    return (float *)ptr;
}

static void zero_matrix(int M, int N, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        memset(C + (size_t)i * (size_t)ldc, 0, (size_t)N * sizeof(float));
    }
}

static void scale_matrix(int M, int N, float *C, int ldc, float beta) {
    for (int i = 0; i < M; ++i) {
        float *row = C + (size_t)i * (size_t)ldc;
        for (int j = 0; j < N; ++j) {
            row[j] *= beta;
        }
    }
}

static void pack_a_block(int mc, int kc, const float *A, int lda, float *pack) {
    for (int p = 0; p < kc; ++p) {
        const float *a_col = A + p;
        for (int i = 0; i < mc; ++i) {
            pack[p * mc + i] = a_col[i * lda];
        }
    }
}

static void pack_b_block(int kc, int nc, const float *B, int ldb, float *pack) {
    for (int p = 0; p < kc; ++p) {
        const float *b_row = B + (size_t)p * (size_t)ldb;
        memcpy(pack + (size_t)p * (size_t)nc, b_row, (size_t)nc * sizeof(float));
    }
}

static inline void kernel_scalar(int mr,
                                 int nr,
                                 int kc,
                                 const float *packA,
                                 int a_stride,
                                 const float *packB,
                                 int b_stride,
                                 float *C,
                                 int ldc,
                                 float alpha,
                                 float beta) {
    for (int ii = 0; ii < mr; ++ii) {
        for (int jj = 0; jj < nr; ++jj) {
            float acc = 0.0f;
            for (int p = 0; p < kc; ++p) {
                acc += packA[(size_t)p * (size_t)a_stride + ii] *
                       packB[(size_t)p * (size_t)b_stride + jj];
            }
            float value = alpha * acc;
            if (beta != 0.0f) {
                value += beta * C[(size_t)ii * (size_t)ldc + jj];
            }
            C[(size_t)ii * (size_t)ldc + jj] = value;
        }
    }
}

static inline void kernel_4x4(int kc,
                              const float *packA,
                              int a_stride,
                              const float *packB,
                              int b_stride,
                              float *C,
                              int ldc,
                              float alpha,
                              float beta) {
    float32x4_t c0 = vdupq_n_f32(0.0f);
    float32x4_t c1 = vdupq_n_f32(0.0f);
    float32x4_t c2 = vdupq_n_f32(0.0f);
    float32x4_t c3 = vdupq_n_f32(0.0f);

    for (int p = 0; p < kc; ++p) {
        const float *a_row = packA + (size_t)p * (size_t)a_stride;
        const float *b_row = packB + (size_t)p * (size_t)b_stride;

        float32x4_t bvec = vld1q_f32(b_row);

        float32x4_t a0 = vdupq_n_f32(a_row[0]);
        float32x4_t a1 = vdupq_n_f32(a_row[1]);
        float32x4_t a2 = vdupq_n_f32(a_row[2]);
        float32x4_t a3 = vdupq_n_f32(a_row[3]);

        c0 = vfmaq_f32(c0, bvec, a0);
        c1 = vfmaq_f32(c1, bvec, a1);
        c2 = vfmaq_f32(c2, bvec, a2);
        c3 = vfmaq_f32(c3, bvec, a3);
    }

    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    float32x4_t beta_vec = vdupq_n_f32(beta);

    float *c0_ptr = C;
    float *c1_ptr = C + ldc;
    float *c2_ptr = C + 2 * ldc;
    float *c3_ptr = C + 3 * ldc;

    if (beta != 0.0f) {
        float32x4_t c0_orig = vld1q_f32(c0_ptr);
        float32x4_t c1_orig = vld1q_f32(c1_ptr);
        float32x4_t c2_orig = vld1q_f32(c2_ptr);
        float32x4_t c3_orig = vld1q_f32(c3_ptr);

        c0 = vfmaq_f32(vmulq_f32(c0, alpha_vec), c0_orig, beta_vec);
        c1 = vfmaq_f32(vmulq_f32(c1, alpha_vec), c1_orig, beta_vec);
        c2 = vfmaq_f32(vmulq_f32(c2, alpha_vec), c2_orig, beta_vec);
        c3 = vfmaq_f32(vmulq_f32(c3, alpha_vec), c3_orig, beta_vec);
    } else {
        c0 = vmulq_f32(c0, alpha_vec);
        c1 = vmulq_f32(c1, alpha_vec);
        c2 = vmulq_f32(c2, alpha_vec);
        c3 = vmulq_f32(c3, alpha_vec);
    }

    vst1q_f32(c0_ptr, c0);
    vst1q_f32(c1_ptr, c1);
    vst1q_f32(c2_ptr, c2);
    vst1q_f32(c3_ptr, c3);
}

static void compute_block(int mc,
                          int nc,
                          int kc,
                          float alpha,
                          float beta,
                          float *packA,
                          float *packB,
                          float *C,
                          int ldc) {
    int i = 0;
    for (; i + SIMPLE_MR <= mc; i += SIMPLE_MR) {
        int j = 0;
        for (; j + SIMPLE_NR <= nc; j += SIMPLE_NR) {
            kernel_4x4(kc,
                       packA + i,
                       mc,
                       packB + j,
                       nc,
                       C + (size_t)i * (size_t)ldc + j,
                       ldc,
                       alpha,
                       beta);
        }
        if (j < nc) {
            int nr = nc - j;
            kernel_scalar(SIMPLE_MR,
                          nr,
                          kc,
                          packA + i,
                          mc,
                          packB + j,
                          nc,
                          C + (size_t)i * (size_t)ldc + j,
                          ldc,
                          alpha,
                          beta);
        }
    }

    if (i < mc) {
        int mr = mc - i;
        int j = 0;
        for (; j + SIMPLE_NR <= nc; j += SIMPLE_NR) {
            kernel_scalar(mr,
                          SIMPLE_NR,
                          kc,
                          packA + i,
                          mc,
                          packB + j,
                          nc,
                          C + (size_t)i * (size_t)ldc + j,
                          ldc,
                          alpha,
                          beta);
        }
        if (j < nc) {
            int nr = nc - j;
            kernel_scalar(mr,
                          nr,
                          kc,
                          packA + i,
                          mc,
                          packB + j,
                          nc,
                          C + (size_t)i * (size_t)ldc + j,
                          ldc,
                          alpha,
                          beta);
        }
    }
}

static bool sgemm_arm64_nn(int M,
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
    float *packA = simple_aligned_alloc((size_t)SIMPLE_MC * (size_t)SIMPLE_KC);
    float *packB = simple_aligned_alloc((size_t)SIMPLE_KC * (size_t)SIMPLE_NC);
    if (!packA || !packB) {
        free(packA);
        free(packB);
        return false;
    }

    if (alpha == 0.0f) {
        if (beta == 0.0f) {
            zero_matrix(M, N, C, ldc);
        } else if (beta != 1.0f) {
            scale_matrix(M, N, C, ldc, beta);
        }
        free(packA);
        free(packB);
        return true;
    }

    float kernel_beta = 1.0f;
    if (beta == 0.0f) {
        zero_matrix(M, N, C, ldc);
        kernel_beta = 0.0f;
    } else if (beta != 1.0f) {
        scale_matrix(M, N, C, ldc, beta);
    }

    for (int jc = 0; jc < N; jc += SIMPLE_NC) {
        int nc = simple_min_int(SIMPLE_NC, N - jc);
        for (int pc = 0; pc < K; pc += SIMPLE_KC) {
            int kc = simple_min_int(SIMPLE_KC, K - pc);
            const float *B_block = B + (size_t)pc * (size_t)ldb + jc;
            pack_b_block(kc, nc, B_block, ldb, packB);

            for (int ic = 0; ic < M; ic += SIMPLE_MC) {
                int mc = simple_min_int(SIMPLE_MC, M - ic);
                const float *A_block = A + (size_t)ic * (size_t)lda + pc;
                pack_a_block(mc, kc, A_block, lda, packA);

                float *C_block = C + (size_t)ic * (size_t)ldc + jc;
                float panel_beta = (pc == 0) ? kernel_beta : 1.0f;
                compute_block(mc, nc, kc, alpha, panel_beta, packA, packB, C_block, ldc);
            }
        }
    }

    free(packA);
    free(packB);
    return true;
}
#endif /* SIMPLE_BLAS_ARM64 */

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

#if SIMPLE_BLAS_ARM64
    if (row_major && !trans_a && !trans_b) {
        if (sgemm_arm64_nn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)) {
            return;
        }
    }
#endif

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
            if (beta == 0.0f) {
                C[c_idx] = alpha * acc;
            } else {
                C[c_idx] = alpha * acc + beta * C[c_idx];
            }
        }
    }
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
