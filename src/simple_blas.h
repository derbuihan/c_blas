// SPDX-License-Identifier: MIT
#ifndef SIMPLE_BLAS_H
#define SIMPLE_BLAS_H

#include <stddef.h>

#if !(defined(SIMPLE_BLAS_SKIP_SYSTEM_CBLAS))
#if defined(__has_include)
#if __has_include(<cblas.h>)
#include <cblas.h>
#define SIMPLE_BLAS_HAS_CBLAS_TYPES 1
#endif
#endif
#endif

#ifdef CBLAS_H
#define SIMPLE_BLAS_HAS_CBLAS_TYPES 1
#endif

#ifndef SIMPLE_BLAS_HAS_CBLAS_TYPES
/* Minimal CBLAS enum definitions for standalone builds. */
enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
};

enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};
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
                        int ldc);

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
                 int ldc);
#endif /* SIMPLE_BLAS_NO_ALIAS */

#endif /* SIMPLE_BLAS_H */
