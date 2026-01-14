# c_blas

`c_blas` is a compact SGEMM implementation written in C11 with an ARM64-optimized fast path. It exposes the familiar `cblas_sgemm`-compatible API while remaining dependency-free so you can drop it into small projects or benchmarking setups.

## Quick Start

```bash
make bench      # build the library + benchmark driver
make run        # run GFLOP/s benchmarks (uses SIMPLE_SGEMM_THREADS if set)
make test       # run the correctness tests in tests/test_sgemm.c
make clean      # remove build artifacts
```

## Features

- **Reference kernel** for correctness plus an **ARM64 NEON microkernel** with cache-blocking and multi-threading (tuned for Apple Silicon and other aarch64 hosts).
- **Benchmark harness** (`./bench`) that compares `simple` against dynamically loaded BLAS backends (OpenBLAS on Linux, OpenBLAS/Accelerate on macOS) and prints median GFLOP/s.
- **CBLAS shim** so existing BLAS callers can link against `simple_blas` without code changes.

## Configuration Tips

- Set `SIMPLE_SGEMM_THREADS` to cap the threaded fast path (CI uses `2` threads on small arm instances).
- Override BLAS library paths with `OPENBLAS_DYLIB`, `OPENBLAS_LIB`, or `BLAS_LIB` when the defaults aren’t picked up.
- Use `CC=clang make run` or customize `CFLAGS` if you need different toolchains.

See `AGENTS.md` or `GEMINI.md` for deeper project guidelines and architecture notes.
