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

#define MT_THRESHOLD_M 256
#define MAX_THREADS 16

static void simple_blas_select_blocking(int M, int N, int K, int *mc, int *nc, int *kc) {
    int min_mn = (M < N) ? M : N;
    int mc_val = MC;
    int nc_val = NC;
    int kc_val = KC;

    if (min_mn <= 192) {
        mc_val = 128;
        nc_val = 256;
        kc_val = 128;
    } else if (min_mn <= 384) {
        mc_val = 176;
        nc_val = 320;
        kc_val = 160;
    } else if (min_mn <= 768) {
        mc_val = 208;
        nc_val = 384;
        kc_val = 192;
    } else if (min_mn <= 1536) {
        mc_val = 256;
        nc_val = 512;
        kc_val = 224;
    } else {
        mc_val = 320;
        nc_val = 512;
        kc_val = 256;
    }

    if (mc_val > M) mc_val = M;
    if (nc_val > N) nc_val = N;
    if (kc_val > K) kc_val = K;

    if (mc_val < SIMPLE_BLAS_ARM64_TILE_M) mc_val = SIMPLE_BLAS_ARM64_TILE_M;
    if (nc_val < SIMPLE_BLAS_ARM64_TILE_N) nc_val = SIMPLE_BLAS_ARM64_TILE_N;
    if (kc_val < SIMPLE_BLAS_ARM64_TILE_M) kc_val = SIMPLE_BLAS_ARM64_TILE_M;

    *mc = mc_val;
    *nc = nc_val;
    *kc = kc_val;
}

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

static void arm64_compute_block(int row_start,
                                int rows,
                                int col_start,
                                int cols,
                                int pb,
                                int p,
                                float alpha,
                                float beta,
                                const float *A,
                                int lda,
                                const float *B,
                                int ldb,
                                float *C,
                                int ldc,
                                float *packed_A,
                                const float *packed_B) {
    (void)B;
    (void)ldb;
    pack_A(rows, pb, A + (size_t)row_start * lda + p, lda, packed_A);
    for (int jr = 0; jr < cols; jr += SIMPLE_BLAS_ARM64_TILE_N) {
        for (int ir = 0; ir < rows; ir += SIMPLE_BLAS_ARM64_TILE_M) {
            float *ptr_c = C + (size_t)(row_start + ir) * ldc + (col_start + jr);
            if (ir + SIMPLE_BLAS_ARM64_TILE_M <= rows && jr + SIMPLE_BLAS_ARM64_TILE_N <= cols) {
                simple_blas_arm64_kernel_12x8(packed_A + (size_t)ir * pb,
                                              packed_B + (size_t)jr * pb,
                                              ptr_c,
                                              0,
                                              SIMPLE_BLAS_ARM64_TILE_N,
                                              ldc,
                                              pb,
                                              alpha,
                                              beta);
            } else {
                int me = (rows - ir > SIMPLE_BLAS_ARM64_TILE_M) ? SIMPLE_BLAS_ARM64_TILE_M : rows - ir;
                int ne = (cols - jr > SIMPLE_BLAS_ARM64_TILE_N) ? SIMPLE_BLAS_ARM64_TILE_N : cols - jr;
                float tail_block[SIMPLE_BLAS_ARM64_TILE_M * SIMPLE_BLAS_ARM64_TILE_N];
                for (int tr = 0; tr < SIMPLE_BLAS_ARM64_TILE_M; ++tr) {
                    float *dst_row = tail_block + tr * SIMPLE_BLAS_ARM64_TILE_N;
                    if (tr < me) {
                        const float *src_row = C + (size_t)(row_start + ir + tr) * ldc + (col_start + jr);
                        for (int tc = 0; tc < SIMPLE_BLAS_ARM64_TILE_N; ++tc) {
                            if (tc < ne) {
                                dst_row[tc] = src_row[tc];
                            } else {
                                dst_row[tc] = 0.0f;
                            }
                        }
                    } else {
                        for (int tc = 0; tc < SIMPLE_BLAS_ARM64_TILE_N; ++tc) {
                            dst_row[tc] = 0.0f;
                        }
                    }
                }
                simple_blas_arm64_kernel_12x8(packed_A + (size_t)ir * pb,
                                              packed_B + (size_t)jr * pb,
                                              tail_block,
                                              0,
                                              SIMPLE_BLAS_ARM64_TILE_N,
                                              SIMPLE_BLAS_ARM64_TILE_N,
                                              pb,
                                              alpha,
                                              beta);
                for (int tr = 0; tr < me; ++tr) {
                    float *dst_row = C + (size_t)(row_start + ir + tr) * ldc + (col_start + jr);
                    const float *src_row = tail_block + tr * SIMPLE_BLAS_ARM64_TILE_N;
                    for (int tc = 0; tc < ne; ++tc) {
                        dst_row[tc] = src_row[tc];
                    }
                }
            }
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
    int mc = MC, nc = NC, kc = KC;
    simple_blas_select_blocking(M, N, K, &mc, &nc, &kc);
    int mc_storage = ((mc + SIMPLE_BLAS_ARM64_TILE_M - 1) / SIMPLE_BLAS_ARM64_TILE_M) * SIMPLE_BLAS_ARM64_TILE_M;
    int nc_storage = ((nc + SIMPLE_BLAS_ARM64_TILE_N - 1) / SIMPLE_BLAS_ARM64_TILE_N) * SIMPLE_BLAS_ARM64_TILE_N;

    size_t size_A = (size_t)mc_storage * (size_t)kc * sizeof(float);
    size_t size_B = (size_t)kc * (size_t)nc_storage * sizeof(float);
    float *packed_A = malloc(size_A);
    float *packed_B = malloc(size_B);
    if (!packed_A || !packed_B) {
        free(packed_A); free(packed_B);
        reference_sgemm(true, false, false, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    for (int j = 0; j < N; j += nc) {
        int jb = (N - j > nc) ? nc : N - j;
        for (int p = 0; p < K; p += kc) {
            int pb = (K - p > kc) ? kc : K - p;
            pack_B(pb, jb, B + (size_t)p * ldb + j, ldb, packed_B);
            float current_beta = (p == 0) ? beta : 1.0f;
            for (int i = 0; i < M; i += mc) {
                int ib = (M - i > mc) ? mc : M - i;
                arm64_compute_block(i,
                                    ib,
                                    j,
                                    jb,
                                    pb,
                                    p,
                                    alpha,
                                    current_beta,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    C,
                                    ldc,
                                    packed_A,
                                    packed_B);
            }
        }
    }
    free(packed_A); free(packed_B);
}

typedef struct {
    int thread_id;
    int thread_count;
    int blocks;
    int mc;
    int pb;
    int jb;
    int j;
    int p;
    int M;
    float alpha;
    float beta;
    const float *A;
    const float *B;
    float *C;
    int lda;
    int ldb;
    int ldc;
    float *packed_A;
    const float *packed_B;
} Arm64ThreadArgs;

static void *arm64_sgemm_block_worker(void *arg) {
    Arm64ThreadArgs *w = (Arm64ThreadArgs *)arg;
    for (int block = w->thread_id; block < w->blocks; block += w->thread_count) {
        int i = block * w->mc;
        if (i >= w->M) break;
        int ib = (w->M - i > w->mc) ? w->mc : w->M - i;
        arm64_compute_block(i,
                            ib,
                            w->j,
                            w->jb,
                            w->pb,
                            w->p,
                            w->alpha,
                            w->beta,
                            w->A,
                            w->lda,
                            w->B,
                            w->ldb,
                            w->C,
                            w->ldc,
                            w->packed_A,
                            w->packed_B);
    }
    return NULL;
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

static void arm64_row_major_sgemm_threaded(int M,
                                           int N,
                                           int K,
                                           float alpha,
                                           const float *A,
                                           int lda,
                                           const float *B,
                                           int ldb,
                                           float beta,
                                           float *C,
                                           int ldc,
                                           int thread_count) {
    if (thread_count <= 1) {
        arm64_row_major_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    int mc = MC, nc = NC, kc = KC;
    simple_blas_select_blocking(M, N, K, &mc, &nc, &kc);
    int mc_storage = ((mc + SIMPLE_BLAS_ARM64_TILE_M - 1) / SIMPLE_BLAS_ARM64_TILE_M) * SIMPLE_BLAS_ARM64_TILE_M;
    int nc_storage = ((nc + SIMPLE_BLAS_ARM64_TILE_N - 1) / SIMPLE_BLAS_ARM64_TILE_N) * SIMPLE_BLAS_ARM64_TILE_N;

    size_t size_A = (size_t)mc_storage * (size_t)kc * sizeof(float);
    size_t size_B = (size_t)kc * (size_t)nc_storage * sizeof(float);
    float *packed_B = malloc(size_B);
    float *packed_A_buffers[MAX_THREADS] = {0};
    if (!packed_B) {
        arm64_row_major_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    int actual_threads = thread_count;
    if (actual_threads > MAX_THREADS) actual_threads = MAX_THREADS;
    for (int t = 0; t < actual_threads; ++t) {
        packed_A_buffers[t] = malloc(size_A);
        if (!packed_A_buffers[t]) {
            for (int i = 0; i < t; ++i) free(packed_A_buffers[i]);
            free(packed_B);
            arm64_row_major_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
    }

    for (int j = 0; j < N; j += nc) {
        int jb = (N - j > nc) ? nc : N - j;
        for (int p = 0; p < K; p += kc) {
            int pb = (K - p > kc) ? kc : K - p;
            pack_B(pb, jb, B + (size_t)p * ldb + j, ldb, packed_B);
            float current_beta = (p == 0) ? beta : 1.0f;
            int blocks = (M + mc - 1) / mc;
            int active_threads = actual_threads;
            if (blocks < active_threads) active_threads = blocks;
            if (active_threads <= 1) {
                for (int i = 0; i < M; i += mc) {
                    int ib = (M - i > mc) ? mc : M - i;
                    arm64_compute_block(i,
                                        ib,
                                        j,
                                        jb,
                                        pb,
                                        p,
                                        alpha,
                                        current_beta,
                                        A,
                                        lda,
                                        B,
                                        ldb,
                                        C,
                                        ldc,
                                        packed_A_buffers[0],
                                        packed_B);
                }
                continue;
            }

            pthread_t threads[MAX_THREADS];
            Arm64ThreadArgs args[MAX_THREADS];
            for (int t = 0; t < active_threads; ++t) {
                args[t] = (Arm64ThreadArgs){
                    .thread_id = t,
                    .thread_count = active_threads,
                    .blocks = blocks,
                    .mc = mc,
                    .pb = pb,
                    .jb = jb,
                    .j = j,
                    .p = p,
                    .M = M,
                    .alpha = alpha,
                    .beta = current_beta,
                    .A = A,
                    .B = B,
                    .C = C,
                    .lda = lda,
                    .ldb = ldb,
                    .ldc = ldc,
                    .packed_A = packed_A_buffers[t],
                    .packed_B = packed_B,
                };
            }
            for (int t = 0; t < active_threads - 1; ++t) {
                pthread_create(&threads[t], NULL, arm64_sgemm_block_worker, &args[t]);
            }
            arm64_sgemm_block_worker(&args[active_threads - 1]);
            for (int t = 0; t < active_threads - 1; ++t) {
                pthread_join(threads[t], NULL);
            }
        }
    }

    for (int t = 0; t < actual_threads; ++t) {
        free(packed_A_buffers[t]);
    }
    free(packed_B);
}

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

static bool arm64_try_fast_path(bool rm, bool ta, bool tb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    if (!rm || ta || tb) return false;
    int nt = simple_blas_thread_cap();
    if (M < MT_THRESHOLD_M || nt <= 1) {
        arm64_row_major_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return true;
    }
    int min_dim = (M < N) ? M : N;
    if (min_dim < 768 || min_dim >= 1536) {
        pthread_t t[MAX_THREADS]; ThreadArgs args[MAX_THREADS];
        if (nt > MAX_THREADS) nt = MAX_THREADS;
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
    arm64_row_major_sgemm_threaded(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, nt);
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
