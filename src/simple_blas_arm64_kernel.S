// SPDX-License-Identifier: MIT

#if defined(__APPLE__)
#define FUNC(name) _##name
#else
#define FUNC(name) name
#endif

    .text
    .p2align 2
    .globl FUNC(simple_blas_arm64_kernel_4x4)
#if !defined(__APPLE__)
    .type FUNC(simple_blas_arm64_kernel_4x4), %function
#endif
FUNC(simple_blas_arm64_kernel_4x4):
    // Preserve alpha/beta for later use.
    fmov    s30, s0
    fmov    s31, s1

    // Convert leading dimensions to byte strides.
    uxtw    x8, w3                 // lda
    lsl     x8, x8, #2
    uxtw    x9, w4                 // ldb
    lsl     x9, x9, #2
    uxtw    x10, w5                // ldc
    lsl     x10, x10, #2

    // Set up row pointers for A.
    mov     x11, x0
    add     x12, x11, x8
    add     x13, x12, x8
    add     x14, x13, x8

    // Pointer to current row of B.
    mov     x15, x1

    // Zero accumulators (use caller-saved v16-v19).
    eor     v16.16b, v16.16b, v16.16b
    eor     v17.16b, v17.16b, v17.16b
    eor     v18.16b, v18.16b, v18.16b
    eor     v19.16b, v19.16b, v19.16b

    mov     w7, w6
    cbz     w7, 2f

1:  ldr     q20, [x15]
    add     x15, x15, x9

    ldr     s0, [x11], #4
    ldr     s1, [x12], #4
    ldr     s2, [x13], #4
    ldr     s3, [x14], #4

    fmla    v16.4s, v20.4s, v0.s[0]
    fmla    v17.4s, v20.4s, v1.s[0]
    fmla    v18.4s, v20.4s, v2.s[0]
    fmla    v19.4s, v20.4s, v3.s[0]

    subs    w7, w7, #1
    b.ne    1b

2:  // Apply alpha.
    dup     v21.4s, v30.s[0]
    fmul    v16.4s, v16.4s, v21.4s
    fmul    v17.4s, v17.4s, v21.4s
    fmul    v18.4s, v18.4s, v21.4s
    fmul    v19.4s, v19.4s, v21.4s

    // Compute C row pointers.
    mov     x11, x2
    add     x12, x11, x10
    add     x13, x12, x10
    add     x14, x13, x10

    // Check beta == 0 (treat +/-0 the same).
    fmov    w7, s31
    and     w7, w7, #0x7fffffff
    cbz     w7, 4f

    // General beta path.
    dup     v22.4s, v31.s[0]

    ldr     q24, [x11]
    ldr     q25, [x12]
    ldr     q26, [x13]
    ldr     q27, [x14]

    fmul    v24.4s, v24.4s, v22.4s
    fmul    v25.4s, v25.4s, v22.4s
    fmul    v26.4s, v26.4s, v22.4s
    fmul    v27.4s, v27.4s, v22.4s

    fadd    v16.4s, v16.4s, v24.4s
    fadd    v17.4s, v17.4s, v25.4s
    fadd    v18.4s, v18.4s, v26.4s
    fadd    v19.4s, v19.4s, v27.4s

4:  str     q16, [x11]
    str     q17, [x12]
    str     q18, [x13]
    str     q19, [x14]
    ret

#if !defined(__APPLE__)
    .size FUNC(simple_blas_arm64_kernel_4x4), .-FUNC(simple_blas_arm64_kernel_4x4)
#endif
