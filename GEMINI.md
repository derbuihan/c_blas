# c_blas

## Project Overview
`c_blas` is a lightweight, standalone implementation of single-precision matrix multiplication (SGEMM) written in C11. It is designed to be a readable, educational, and reasonably performant drop-in replacement for standard BLAS libraries in specific contexts.

**Key Features:**
*   **Reference Implementation:** A clear, naive implementation for correctness verification.
*   **ARM64 Optimization:** specialized kernels using NEON intrinsics, loop unrolling, and cache blocking (tiling) for Apple Silicon and other ARM64 targets.
*   **CBLAS Compatibility:** Exposes an API compatible with the standard `cblas_sgemm` function, allowing it to work as a shim.
*   **Zero Dependencies:** Designed to build without external BLAS libraries, though it can optionally alias standard CBLAS types if headers are present.

## Directory Structure
*   `src/`: Core source code.
    *   `simple_blas.c`: Main logic, including reference implementation and packing routines for the optimized kernels.
    *   `simple_blas.h`: Public API definitions and minimal CBLAS enum shims.
    *   `simple_blas_arm64_kernel.S`: Assembly optimized kernel (linked if on ARM64).
    *   `benchmark.c`: Benchmarking driver to measure GFLOP/s.
*   `tests/`: Correctness tests.
    *   `test_sgemm.c`: Validates the implementation against a naive reference GEMM across various matrix sizes and edge cases.
*   `build/`: Output directory for object files (created during build).
*   `Makefile`: Entry point for building and testing.

## Building and Running

The project uses a standard `Makefile`.

### Core Commands
*   **Build Benchmark:** `make` or `make bench`
    *   Compiles the code with `-O3` and produces the `bench` executable.
*   **Run Benchmark:** `make run`
    *   Builds and executes the benchmark suite.
*   **Run Tests:** `make test`
    *   Builds the test runner (`test_runner`) and executes `tests/test_sgemm.c` to verify correctness.
*   **Clean:** `make clean`
    *   Removes the `build/` directory and executables.

### Environment Variables
*   `CC`: Override the compiler (default: `cc`). Example: `CC=clang make run`.
*   `CFLAGS`: Override compiler flags.
*   `SIMPLE_SGEMM_THREADS`: (Runtime) Controls the number of threads used for the ARM64 fast path.

## Development Conventions

*   **Language:** C11 standard (`-std=c11`).
*   **Formatting:** 4-space indentation, braces on the same line.
*   **Naming:**
    *   Internal functions: `simple_` prefix.
    *   Public API: `simple_cblas_` prefix.
    *   Macros: `SIMPLE_` prefix.
*   **Optimization:**
    *   Use `static inline` for helper functions to encourage inlining.
    *   Guard architecture-specific code (like SIMD) with preprocessor checks (e.g., `#if defined(__aarch64__)`).
    *   Keep error paths lightweight.
*   **Testing:**
    *   Always ensure `make test` passes before submitting changes.
    *   Add new edge cases to `tests/test_sgemm.c` if discovering bugs.
*   **License:** Includes SPDX-License-Identifier: MIT at the top of files.
