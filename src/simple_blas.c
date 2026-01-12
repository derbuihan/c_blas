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
void simple_blas_arm64_kernel_12x8(const float *A,
                                   const float *B,
                                   float *C,
                                   int lda,
                                   int ldb,
                                   int ldc,
                                   int K,
                                   float alpha,
                                   float beta);
#define SIMPLE_BLAS_ARM64_TILE_M 12
#define SIMPLE_BLAS_ARM64_TILE_N 8

// Cache blocking parameters (tuned for Apple Silicon / ARM64)
// MC must be a multiple of TILE_M (12) to handle packing padding correctly without buffer overflow
#define MC 264
#define KC 256
#define NC 512

static void pack_A(int M, int K, const float *A, int lda, float *buffer) {
    int i = 0;
    for (; i + SIMPLE_BLAS_ARM64_TILE_M <= M; i += SIMPLE_BLAS_ARM64_TILE_M) {
        for (int p = 0; p < K; ++p) {
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
        for (int p = 0; p < K; ++p) {
            const float *b_ptr = B + (size_t)p * ldb + j;
            for (int k = 0; k < SIMPLE_BLAS_ARM64_TILE_N; ++k) {
                *buffer++ = b_ptr[k];
            }
        }
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
    // Buffers for packing.
    // MC must be aligned.
    size_t size_A = (size_t)MC * (size_t)KC * sizeof(float);
    size_t size_B = (size_t)KC * (size_t)NC * sizeof(float);
    
    float *packed_A = malloc(size_A);
    float *packed_B = malloc(size_B);
    
    if (!packed_A || !packed_B) {
        free(packed_A);
        free(packed_B);
        reference_sgemm(true, false, false, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    for (int j = 0; j < N; j += NC) {
        int jb = N - j;
        if (jb > NC) jb = NC;
        
        for (int p = 0; p < K; p += KC) {
            int pb = K - p;
            if (pb > KC) pb = KC;
            
            // Pack B: sub-panel B[p..p+pb][j..j+jb]
            pack_B(pb, jb, B + (size_t)p * ldb + j, ldb, packed_B);

            for (int i = 0; i < M; i += MC) {
                int ib = M - i;
                if (ib > MC) ib = MC;
                
                // Pack A: sub-panel A[i..i+ib][p..p+pb]
                pack_A(ib, pb, A + (size_t)i * lda + p, lda, packed_A);
                
                for (int jr = 0; jr < jb; jr += SIMPLE_BLAS_ARM64_TILE_N) {
                    for (int ir = 0; ir < ib; ir += SIMPLE_BLAS_ARM64_TILE_M) {
                        
                        const float *ptr_a = packed_A + (size_t)ir * pb;
                        const float *ptr_b = packed_B + (size_t)jr * pb;
                        float *ptr_c = C + (size_t)(i + ir) * ldc + (j + jr);
                        
                        if (ir + SIMPLE_BLAS_ARM64_TILE_M <= ib && jr + SIMPLE_BLAS_ARM64_TILE_N <= jb) {
                            float current_beta = (p == 0) ? beta : 1.0f;
                            simple_blas_arm64_kernel_12x8(ptr_a,
                                                          ptr_b,
                                                          ptr_c,
                                                          0,
                                                          SIMPLE_BLAS_ARM64_TILE_N,
                                                          ldc,
                                                          pb,
                                                          alpha,
                                                          current_beta);
                        } else {
                            float current_beta = (p == 0) ? beta : 1.0f;
                            int m_edge = (ib - ir);
                            if (m_edge > SIMPLE_BLAS_ARM64_TILE_M) m_edge = SIMPLE_BLAS_ARM64_TILE_M;
                            
                            int n_edge = (jb - jr);
                            if (n_edge > SIMPLE_BLAS_ARM64_TILE_N) n_edge = SIMPLE_BLAS_ARM64_TILE_N;
                            
                            scalar_tail_block(i + ir,
                                              j + jr,
                                              m_edge,
                                              n_edge,
                                              pb,
                                              alpha,
                                              A + (size_t)p, 
                                              lda,
                                              B + (size_t)p * ldb, 
                                              ldb,
                                              current_beta,
                                              C,
                                              ldc);
                        }
                    }
                }
            }
        }
    }
    
    free(packed_A);
    free(packed_B);
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