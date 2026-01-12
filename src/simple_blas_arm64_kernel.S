// SPDX-License-Identifier: MIT

#if defined(__APPLE__)
#define FUNC(name) _##name
#else
#define FUNC(name) name
#endif

    .text
    .p2align 2
    .globl FUNC(simple_blas_arm64_kernel_12x8)
#if !defined(__APPLE__)
    .type FUNC(simple_blas_arm64_kernel_12x8), %function
#endif
FUNC(simple_blas_arm64_kernel_12x8):
    // x0: packed_A, x1: B, x2: C, w3: lda (ignored), w4: ldb, w5: ldc, w6: K, s0: alpha, s1: beta

    // Save callee-saved registers and alpha/beta.
    sub     sp, sp, #128
    stp     d8, d9, [sp, #0]
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #64]
    stp     x21, x22, [sp, #80]
    str     x23, [sp, #96]
    stp     s0, s1, [sp, #112]

    // Convert leading dimensions to byte strides.
    uxtw    x9, w4                 // ldb
    lsl     x9, x9, #2
    uxtw    x10, w5                // ldc
    lsl     x10, x10, #2

    // Zero accumulators v8-v31.
    .irp v,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    eor     v\v\().16b, v\v\().16b, v\v\().16b
    .endr

    cbz     w6, 4f
    lsr     w7, w6, #2      // K / 4
    cbz     w7, 2f

1:  // Unrolled K loop (x4) with packed A.
    .irp p,0,1,2,3
    ldr     q0, [x1]        // B[p][j...j+3]
    ldr     q1, [x1, #16]   // B[p][j+4...j+7]
    add     x1, x1, x9

    ldr     q2, [x0]        // A[i...i+3][p]
    ldr     q3, [x0, #16]   // A[i+4...i+7][p]
    ldr     q4, [x0, #32]   // A[i+8...i+11][p]
    add     x0, x0, #48

    fmla    v8.4s,  v0.4s, v2.s[0]
    fmla    v9.4s,  v1.4s, v2.s[0]
    fmla    v10.4s, v0.4s, v2.s[1]
    fmla    v11.4s, v1.4s, v2.s[1]
    fmla    v12.4s, v0.4s, v2.s[2]
    fmla    v13.4s, v1.4s, v2.s[2]
    fmla    v14.4s, v0.4s, v2.s[3]
    fmla    v15.4s, v1.4s, v2.s[3]

    fmla    v16.4s, v0.4s, v3.s[0]
    fmla    v17.4s, v1.4s, v3.s[0]
    fmla    v18.4s, v0.4s, v3.s[1]
    fmla    v19.4s, v1.4s, v3.s[1]
    fmla    v20.4s, v0.4s, v3.s[2]
    fmla    v21.4s, v1.4s, v3.s[2]
    fmla    v22.4s, v0.4s, v3.s[3]
    fmla    v23.4s, v1.4s, v3.s[3]

    fmla    v24.4s, v0.4s, v4.s[0]
    fmla    v25.4s, v1.4s, v4.s[0]
    fmla    v26.4s, v0.4s, v4.s[1]
    fmla    v27.4s, v1.4s, v4.s[1]
    fmla    v28.4s, v0.4s, v4.s[2]
    fmla    v29.4s, v1.4s, v4.s[2]
    fmla    v30.4s, v0.4s, v4.s[3]
    fmla    v31.4s, v1.4s, v4.s[3]
    .endr

    subs    w7, w7, #1
    b.ne    1b

2:  ands    w7, w6, #3      // K % 4
    cbz     w7, 4f

3:  // Tail K loop.
    ldr     q0, [x1]
    ldr     q1, [x1, #16]
    add     x1, x1, x9

    ldr     q2, [x0]
    ldr     q3, [x0, #16]
    ldr     q4, [x0, #32]
    add     x0, x0, #48

    fmla    v8.4s,  v0.4s, v2.s[0]
    fmla    v9.4s,  v1.4s, v2.s[0]
    fmla    v10.4s, v0.4s, v2.s[1]
    fmla    v11.4s, v1.4s, v2.s[1]
    fmla    v12.4s, v0.4s, v2.s[2]
    fmla    v13.4s, v1.4s, v2.s[2]
    fmla    v14.4s, v0.4s, v2.s[3]
    fmla    v15.4s, v1.4s, v2.s[3]

    fmla    v16.4s, v0.4s, v3.s[0]
    fmla    v17.4s, v1.4s, v3.s[0]
    fmla    v18.4s, v0.4s, v3.s[1]
    fmla    v19.4s, v1.4s, v3.s[1]
    fmla    v20.4s, v0.4s, v3.s[2]
    fmla    v21.4s, v1.4s, v3.s[2]
    fmla    v22.4s, v0.4s, v3.s[3]
    fmla    v23.4s, v1.4s, v3.s[3]

    fmla    v24.4s, v0.4s, v4.s[0]
    fmla    v25.4s, v1.4s, v4.s[0]
    fmla    v26.4s, v0.4s, v4.s[1]
    fmla    v27.4s, v1.4s, v4.s[1]
    fmla    v28.4s, v0.4s, v4.s[2]
    fmla    v29.4s, v1.4s, v4.s[2]
    fmla    v30.4s, v0.4s, v4.s[3]
    fmla    v31.4s, v1.4s, v4.s[3]

    subs    w7, w7, #1
    b.ne    3b

4:  // Reload alpha and beta.
    ldp     s0, s1, [sp, #112]

    // Apply alpha.
    dup     v2.4s, v0.s[0]
    .irp v,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    fmul    v\v\().4s, v\v\().4s, v2.4s
    .endr

    // Check beta == 0.
    fmov    w7, s1
    and     w7, w7, #0x7fffffff
    cbz     w7, 6f

    // General beta path.
    dup     v3.4s, v1.s[0]

    .macro LOAD_FMA_STORE_C lo, hi, ptr
    ldr     q6, [\ptr]
    ldr     q7, [\ptr, #16]
    fmla    v\lo\().4s, v6.4s, v3.4s
    fmla    v\hi\().4s, v7.4s, v3.4s
    str     q\lo\(), [\ptr]
    str     q\hi\(), [\ptr, #16]
    .endm

    mov     x11, x2
    add     x12, x11, x10
    add     x13, x12, x10
    add     x14, x13, x10
    add     x15, x14, x10
    add     x16, x15, x10
    add     x17, x16, x10
    add     x19, x17, x10
    add     x20, x19, x10
    add     x21, x20, x10
    add     x22, x21, x10
    add     x23, x22, x10

    LOAD_FMA_STORE_C 8,  9,  x11
    LOAD_FMA_STORE_C 10, 11, x12
    LOAD_FMA_STORE_C 12, 13, x13
    LOAD_FMA_STORE_C 14, 15, x14
    LOAD_FMA_STORE_C 16, 17, x15
    LOAD_FMA_STORE_C 18, 19, x16
    LOAD_FMA_STORE_C 20, 21, x17
    LOAD_FMA_STORE_C 22, 23, x19
    LOAD_FMA_STORE_C 24, 25, x20
    LOAD_FMA_STORE_C 26, 27, x21
    LOAD_FMA_STORE_C 28, 29, x22
    LOAD_FMA_STORE_C 30, 31, x23

    b       7f

6:  // beta == 0: Just store results.
    mov     x11, x2
    add     x12, x11, x10
    add     x13, x12, x10
    add     x14, x13, x10
    add     x15, x14, x10
    add     x16, x15, x10
    add     x17, x16, x10
    add     x19, x17, x10
    add     x20, x19, x10
    add     x21, x20, x10
    add     x22, x21, x10
    add     x23, x22, x10

    str     q8,  [x11]
    str     q9,  [x11, #16]
    str     q10, [x12]
    str     q11, [x12, #16]
    str     q12, [x13]
    str     q13, [x13, #16]
    str     q14, [x14]
    str     q15, [x14, #16]
    str     q16, [x15]
    str     q17, [x15, #16]
    str     q18, [x16]
    str     q19, [x16, #16]
    str     q20, [x17]
    str     q21, [x17, #16]
    str     q22, [x19]
    str     q23, [x19, #16]
    str     q24, [x20]
    str     q25, [x20, #16]
    str     q26, [x21]
    str     q27, [x21, #16]
    str     q28, [x22]
    str     q29, [x22, #16]
    str     q30, [x23]
    str     q31, [x23, #16]

7:  // Restore callee-saved registers and return.
    ldp     d8, d9, [sp, #0]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     x19, x20, [sp, #64]
    ldp     x21, x22, [sp, #80]
    ldr     x23, [sp, #96]
    add     sp, sp, #128
    ret

#if !defined(__APPLE__)
    .size FUNC(simple_blas_arm64_kernel_12x8), .-FUNC(simple_blas_arm64_kernel_12x8)
#endif
