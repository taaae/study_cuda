# CUDA Study Course

A hands-on, increasing-difficulty CUDA course meant to be cloned into **Google Colab (free T4 GPU, `sm_75`)** and worked through one exercise at a time. Every exercise teaches a new concept, is graded for **both correctness and efficiency**, and is written in modern, non-legacy CUDA.

## How to use it in Colab

1. Pick a GPU runtime: *Runtime → Change runtime type → T4 GPU*.
2. Clone the repo and `cd` into an exercise folder:
   ```python
   !git clone <your-repo-url> study_cuda
   %cd study_cuda/01_vector_add
   ```
3. Open the `.cu` file, read `README.md`, and fill in the `TODO`s. Type the CUDA-specific syntax yourself — that is the point.
4. Grade your work:
   ```python
   !python grade.py
   ```
5. Stuck? Open `hints.md` and reveal hints one level at a time.
6. Want to see a worked answer? `solutions/` holds a reference implementation.

## The contract (same for every exercise)

So that grading is uniform, every exercise follows one structure:

| File | Who edits it | Purpose |
|------|--------------|---------|
| `README.md` | — | Concept, task, and the exact syntax you'll need |
| `hints.md` | — | Progressive `<details>` hints, each labeled with what it reveals |
| `<name>.cu` | **you** | Kernel(s) + a `solve(...)` host entry point. This is all you type. |
| `harness.cu` | — | `main()`: builds data, runs `solve`, checks correctness, times it, prints metrics. **Do not edit.** |
| `grade.py` | — | Compiles `harness.cu`, runs it, checks correctness + efficiency thresholds + static source checks |
| `solutions/<name>.cu` | — | Reference solution (drop-in replacement for `<name>.cu`) |

The harness `#include`s your `.cu` via a `SOLUTION_FILE` macro and calls the `solve(...)` function whose signature is documented in each `README.md`. You never write `main()`, data setup, or timing boilerplate — only the GPU logic.

## How efficiency is graded (and why no Nsight)

Nsight Compute's hardware counters are **blocked on free Colab** (no performance-counter permission), so `ncu`-style transaction counts aren't reliable there. Instead the harness measures **achieved memory bandwidth (GB/s)** and/or **GFLOP/s** with CUDA events and compares them against:

- a fraction of the **device's theoretical peak** (queried at runtime, so it adapts to your GPU), and/or
- a required **speedup over a naive baseline** the harness also runs.

This is a faithful proxy: an uncoalesced or bank-conflicting kernel simply cannot hit a high fraction of peak bandwidth. On top of that, `grade.py` does **static source checks** (e.g. "this kernel must use `__shfl_down_sync`", "this one must not call `atomicAdd`") so you can't pass an efficiency exercise with the wrong technique.

> If you later run on a GPU where `ncu` *is* allowed, the relevant exercises mention the exact metrics worth inspecting.

## Curriculum

**Phase 0 — Fundamentals**
1. `01_vector_add` — threads/blocks/grid, indexing, memory model, launch syntax
2. `02_grid_stride` — grid-stride loops, error-checking macro, arbitrary sizes
3. `03_benchmarking` — `cudaEvent` timing, achieved vs theoretical bandwidth

**Phase 1 — Memory hierarchy**
4. `04_coalescing` — global-memory coalescing via matrix transpose
5. `05_shared_transpose` — shared memory + bank-conflict avoidance
6. `06_reduction` — the parallel-reduction optimization ladder
7. `07_warp_reduce` — warp-level primitives (`__shfl_down_sync`)

**Phase 2 — Compute kernels**
8. `08_tiled_gemm` — shared-memory tiled matrix multiply vs cuBLAS
9. `09_gemm_register` — register tiling, thread coarsening, `float4` loads
10. `10_scan` — work-efficient prefix sum
11. `11_histogram` — atomics, shared-memory privatization, contention

**Phase 3 — Tooling & libraries**
12. `12_thrust` — Thrust algorithms and when to reach for them
13. `13_cub` — CUB block/device primitives
14. `14_streams` — pinned memory, streams, copy/compute overlap
15. `15_occupancy` — occupancy API and launch-config tuning

**Phase 4 — Sparse linear algebra**
16. `16_spmv_csr_scalar` — CSR format, one-thread-per-row SpMV
17. `17_spmv_csr_vector` — one-warp-per-row SpMV, load balancing
18. `18_spmv_ell` — ELL/hybrid layout and the load-balancing tradeoff
19. `19_cusparse` — cuSPARSE SpMV/SpMM vs your kernels
20. `20_conjugate_gradient` — capstone: CG solver built on your SpMV

**Phase 5 — Advanced**
21. `21_cooperative_groups` — cooperative groups & grid-wide sync
22. `22_double_buffer` — software pipelining / double-buffered shared memory (the T4-runnable cousin of Ampere `cp.async`)
23. `23_tensor_cores` — WMMA half-precision GEMM (T4 tensor cores)
24. `24_capstone` — optimize an unknown kernel end-to-end

## Prerequisites

Just C and a little linear algebra. Exercise 1 assumes you've never written a line of CUDA.
