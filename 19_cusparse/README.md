# Exercise 19 — cuSPARSE (the modern generic SpMV API)
> You've hand-rolled three SpMV kernels — now call the one NVIDIA spent a decade tuning.

## The idea

You now understand SpMV from the inside: scalar CSR, warp-per-row CSR, and ELL.
That understanding is the whole point of the previous exercises — but in production
you don't ship your own SpMV. You call **cuSPARSE**, NVIDIA's sparse library, whose
kernels pick the layout and launch config per matrix and per GPU. Your hand kernels
exist so you *know what it's doing* and can tell when a custom kernel is actually
worth writing.

This exercise is about **using a library correctly**, which has its own idioms.
You'll call `cusparseSpMV` to compute `y = alpha*A*x + beta*y` for a CSR matrix.

## Under the hood: the generic API design

Old cuSPARSE had one entry point per (format × datatype) — e.g. `cusparseScsrmv`
for single-precision CSR. That combinatorial mess is the **legacy API, and it's
deprecated** — don't use it. The **generic API** separates *describing your data*
from *running the op*, so one `cusparseSpMV` works across CSR/COO/etc and across
float/double/half:

| Object | Created with | Represents |
|--------|--------------|------------|
| handle | `cusparseCreate` | the library context — one per program |
| sparse matrix `A` | `cusparseCreateCsr` | your CSR arrays + dims, as a `cusparseSpMatDescr_t` |
| dense vectors `x`, `y` | `cusparseCreateDnVec` | a device pointer + length, as a `cusparseDnVecDescr_t` |
| the op | `cusparseSpMV` | `y = alpha*A*x + beta*y` |

A *descriptor* is an opaque handle wrapping your pointers plus **type tags** that
tell the library how to read the bytes (32-bit indices? 0- or 1-based? float or
double?). The library never owns or copies your data — the descriptor just points
at your existing device arrays.

### The buffer pattern (the part people miss)

`cusparseSpMV` may need scratch memory, and the amount depends on the matrix and
the chosen algorithm. You don't guess it — you **ask**:

1. `cusparseSpMV_bufferSize(...)` → writes the required `size_t bufferSize`.
2. `cudaMalloc(&dBuffer, bufferSize)` → allocate exactly that (it can be 0).
3. `cusparseSpMV(...)` with that `dBuffer` → runs the multiply.

The query and the run must get the **same** `alpha`, `beta`, descriptors,
operation, compute type, and algorithm — otherwise the size you asked for doesn't
match the call you make. This *query → allocate → execute* dance recurs across cuda
libraries (cuFFT, cuDNN, …); learn it once.

## A picture

```text
  cusparseCreate ──► handle ───────────────────────────────────────────┐
                                                                        │
  rowPtr,colIdx,vals ─► cusparseCreateCsr ─► matA (SpMatDescr) ─┐       │
  x (len ncols)       ─► cusparseCreateDnVec ─► vecX ───────────┤       │
  y (len nrows)       ─► cusparseCreateDnVec ─► vecY ───────────┤       │
                                                                ▼       ▼
                              cusparseSpMV_bufferSize(handle, …, &bufferSize)
                                                                │
                              cudaMalloc(&dBuffer, bufferSize)  │
                                                                ▼
                              cusparseSpMV(handle, …, dBuffer)  ──►  y = A*x
                                                                │
        destroy matA / vecX / vecY,  cudaFree(dBuffer),  cusparseDestroy(handle)
```

## Your task

Compute `y = alpha*A*x + beta*y` for CSR `A` using `cusparseSpMV` with
`CUSPARSE_SPMV_ALG_DEFAULT`. The harness calls your `solve` with `alpha=1, beta=0`
(plain `y = A*x`), but write the general form. Edit `cusparse_spmv.cu`:

1. Create the handle.
2. Create the CSR matrix descriptor and the two dense-vector descriptors.
3. Query the buffer size, then `cudaMalloc` the external buffer.
4. Call `cusparseSpMV`.
5. Destroy the descriptors, free the buffer, destroy the handle.

You do **not** allocate the CSR / x / y device memory — the harness does. A
`CUSPARSE_CHECK` macro is already provided in the file (it works like `CUDA_CHECK`
but for `cusparseStatus_t`), so wrap every cuSPARSE call in it.

### The `solve` contract

```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows, int ncols, int nnz);
```

All pointers are **device pointers**. CSR layout: `rowPtr[nrows+1]`, `colIdx[nnz]`,
`vals[nnz]`. `x` has length `ncols`, `y` has length `nrows`.

## Functions & syntax you'll need

Add `#include <cusparse.h>`. The grader compiles with `-lcusparse`.

| Function | Signature (abbreviated) | Role |
|----------|-------------------------|------|
| `cusparseCreate` | `(cusparseHandle_t* handle)` | make the library context |
| `cusparseCreateCsr` | `(&matA, nrows, ncols, nnz, rowPtr, colIdx, vals, idxType, idxType, idxBase, valType)` | wrap CSR arrays as a sparse descriptor |
| `cusparseCreateDnVec` | `(&vec, size, ptr, valType)` | wrap a device pointer + length as a dense-vector descriptor |
| `cusparseSpMV_bufferSize` | `(handle, op, &alpha, matA, vecX, &beta, vecY, computeType, alg, &bufferSize)` | **query** scratch size into `bufferSize` |
| `cusparseSpMV` | `(handle, op, &alpha, matA, vecX, &beta, vecY, computeType, alg, dBuffer)` | run `y = alpha*A*x + beta*y` |
| `cusparseDestroySpMat` / `cusparseDestroyDnVec` / `cusparseDestroy` | `(descriptor)` / `(handle)` | tear down what you created |
| `cudaMalloc` / `cudaFree` | `(&dBuffer, bufferSize)` / `(dBuffer)` | allocate / free the external buffer |

Key constant values to pass:

- **operation:** `CUSPARSE_OPERATION_NON_TRANSPOSE` (plain `A*x`, not `Aᵀx`)
- **index type:** `CUSPARSE_INDEX_32I` (both `rowPtr` and `colIdx` are 32-bit)
- **index base:** `CUSPARSE_INDEX_BASE_ZERO` (0-based CSR)
- **value / compute type:** `CUDA_R_32F` (real 32-bit float; must match your data)
- **algorithm:** `CUSPARSE_SPMV_ALG_DEFAULT`

Two gotchas: `alpha`/`beta` are passed **by pointer** (`&alpha`, `&beta` — they're
host floats here), and the *compute type* must match the *value type*. **Do not**
call the legacy `cusparseScsrmv` — it's deprecated and the grader rejects it.

## How it's graded

Run `python grade.py` (or `--check-solution` for the reference). Built with
`-lcusparse`.

- **correctness** — `y == A*x` within tolerance, vs a CPU SpMV.
- **efficiency** — the harness times a tiny hand-written scalar CSR kernel and
  reports `speedup_vs_naive`. There is **no hard speedup threshold** — the bar is
  *correct library usage*. It only requires `ms > 0`, i.e. the SpMV actually ran
  and produced a finite timing.
- **source** — you must call `cusparseSpMV`, and must **not** use the deprecated
  `cusparseScsrmv`.

## Going deeper

- **`CUSPARSE_SPMV_ALG_DEFAULT` is a dispatcher.** Under the hood cuSPARSE inspects
  your matrix and picks an implementation — often something like the CSR-vector or
  merge-based kernel. There are also explicit algorithms (`..._CSR_ALG1/ALG2`) you
  can force when profiling.
- **Reuse the buffer.** If you call SpMV repeatedly on the same matrix (as in CG,
  the next exercise), create the descriptors and allocate the buffer *once* and
  reuse them — don't pay the setup per call. Here `solve` is self-contained, so it
  builds and tears down each time, which is exactly what you'd *avoid* in a hot loop.
