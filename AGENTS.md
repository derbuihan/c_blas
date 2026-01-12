# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/`: `simple_blas.c` holds the SGEMM implementation, `simple_blas.h` exposes the API and lightweight CBLAS shims, and `benchmark.c` drives correctness/throughput comparisons. Object files drop into `build/` and the final binary is `bench`. The root `Makefile` is the single entry point; edit it when adding new modules or flags so downstream scripts stay in sync.

## Build, Test, and Development Commands
Use `make` (or `make bench`) to compile with `cc -O3 -std=c11 -Wall -Wextra -pedantic`. `make run` rebuilds if needed and launches `./bench` with the default benchmark suite. `make clean` removes `build/` and the `bench` executable. Override compilers/flags via env vars (`CC=clang make run`) when experimenting with other toolchains.

## Coding Style & Naming Conventions
Stick to C11, 4-space indentation, brace-on-same-line, and keep helpers `static inline` when possible to preserve inlining. Prefix internal symbols with `simple_`, public API with `simple_cblas_*`, and macros/env toggles with `SIMPLE_*`. Retain the MIT SPDX header at the top of every translation unit. Favor explicit `size_t` math, guard SIMD code behind `SIMPLE_BLAS_ARM64`, and keep error paths returning early with informative messages.

## Testing Guidelines
`make run` already seeds matrices, compares against dynamically loaded BLAS backends, and prints GFLOP/s; treat it as both smoke test and performance check. Add new scenarios inside `src/benchmark.c`, keeping naming consistent (`benchmark_backend`). Document every change with the matrix sizes, repetition counts, and acceptable `max_abs_diff` thresholds (<1e-3 today). If you introduce dedicated tests, mirror this structure in a `tests/` subtree and ensure they pass before pushing.

## Commit & Pull Request Guidelines
Follow the existing Conventional Commit style (`refactor(benchmark): ...`, `feat(simple_blas): ...`). Each PR should describe the motivation, summarize functional/perf impact, and include the exact `make run` command plus relevant output or tables. Link issues when available, call out any new env variables (e.g., `SIMPLE_SGEMM_THREADS`, `{OPENBLAS,ACCELERATE}_DYLIB`), and mention how reviewers can reproduce benchmarks locally. Screenshots are unnecessary unless they clarify perf regressions.

## Configuration & Safety Tips
Set `SIMPLE_SGEMM_THREADS` to limit threading; cap it conservatively when benchmarking shared hosts. Use `{OPENBLAS,ACCELERATE}_DYLIB` to point at alternate vendor libraries, and set `SIMPLE_BLAS_DISABLE_ARM64=1` if you need to debug the scalar kernel. Avoid committing compiled artifacts (`bench`, `build/*`) and do not vendor external BLAS libraries—document dependencies instead.
