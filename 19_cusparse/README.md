# Exercise 19 — cuSPARSE (the modern generic SpMV API)

**New concepts:** calling the vendor sparse library **cuSPARSE** through its
**generic API** — opaque matrix/vector *descriptors*, the *buffer-size →
allocate → execute* pattern, and why you reach for a library at all.

## Why a library

You have now hand-written CSR-scalar, CSR-vector, and ELL SpMV. cuSPARSE has spent
years tuning kernels that pick layouts and launch configs per matrix and per GPU.
For production sparse work you call the library; your hand kernels exist so you
*understand* what it's doing and can tell when a custom kernel is worth it.

## The modern generic API

Older cuSPARSE had one function per (format × type), e.g. `cusparseScsrmv`. That
**legacy API is deprecated** — do not use it. The modern **generic API** separates
*describing your data* from *running the op*:

| Object | Created with | Represents |
|--------|--------------|------------|
| handle | `cusparseCreate` | library context (one per program) |
| sparse matrix `A` | `cusparseCreateCsr` | your CSR arrays + dims, as an `cusparseSpMatDescr_t` |
| dense vectors `x`, `y` | `cusparseCreateDnVec` | a device pointer + length, as `cusparseDnVecDescr_t` |
| the op | `cusparseSpMV` | `y = alpha*A*x + beta*y` |

### The buffer pattern (the part people miss)

`cusparseSpMV` may need scratch space. You don't guess its size — you **ask**:

1. `cusparseSpMV_bufferSize(...)` → writes the required `size_t bufferSize`.
2. `cudaMalloc(&dBuffer, bufferSize)` → allocate exactly that (may be 0).
3. `cusparseSpMV(...)` with that `dBuffer` → runs the multiply.

The same `alpha`, `beta`, descriptors, algorithm, and compute type must be passed
to both the `_bufferSize` query and the `cusparseSpMV` call.

## The task

Compute `y = alpha*A*x + beta*y` for a CSR matrix `A` using `cusparseSpMV` with
`CUSPARSE_SPMV_ALG_DEFAULT`. The harness calls your `solve` with `alpha=1, beta=0`
(plain `y = A*x`), but implement the general form. Edit `cusparse_spmv.cu`:

1. Create the handle.
2. Create the CSR matrix descriptor and the two dense-vector descriptors.
3. Query the buffer size, `cudaMalloc` the external buffer.
4. Call `cusparseSpMV`.
5. Destroy the descriptors, free the buffer, destroy the handle.

You do **not** allocate the CSR / x / y device memory — the harness does, and calls
your `solve(...)`.

### `solve` signature (the contract)

```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows, int ncols, int nnz);
```

All pointers are **device pointers**. CSR layout: `rowPtr[nrows+1]`,
`colIdx[nnz]`, `vals[nnz]`. `x` has length `ncols`, `y` has length `nrows`.

## Syntax / reference

Add `#include <cusparse.h>`. Wrap calls in a status check (a `CUSPARSE_CHECK`
macro is fine — analogous to `CUDA_CHECK`). Note the data types:

```cpp
cusparseHandle_t handle;
cusparseCreate(&handle);

cusparseSpMatDescr_t matA;
cusparseDnVecDescr_t vecX, vecY;

// CSR descriptor: note rowPtr/colIdx use 32-bit indices, vals are float.
cusparseCreateCsr(&matA, nrows, ncols, nnz,
                  (void*)rowPtr, (void*)colIdx, (void*)vals,
                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

cusparseCreateDnVec(&vecX, ncols, (void*)x, CUDA_R_32F);
cusparseCreateDnVec(&vecY, nrows, (void*)y, CUDA_R_32F);

float alpha = 1.0f, beta = 0.0f;        // pointers passed below
size_t bufferSize = 0;
cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, &beta, vecY,
                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

void* dBuffer = nullptr;
cudaMalloc(&dBuffer, bufferSize);

cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matA, vecX, &beta, vecY,
             CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

// teardown
cusparseDestroySpMat(matA);
cusparseDestroyDnVec(vecX);
cusparseDestroyDnVec(vecY);
cudaFree(dBuffer);
cusparseDestroy(handle);
```

`alpha` / `beta` are passed **by pointer** (host pointers here). The *compute
type* (`CUDA_R_32F`) must match the value type.

## Grading (`!python grade.py`)

Compiled with `-lcusparse`.

- **correctness** — `y == A*x` within tolerance (vs a CPU SpMV).
- **efficiency** — the harness also runs a tiny hand-written scalar CSR kernel and
  reports `speedup_vs_naive`. There's **no hard speedup threshold** (cuSPARSE
  usually wins on large matrices, but the lesson is *correct library usage*), but
  the SpMV must actually have run.
- **source** — you must call `cusparseSpMV`, and you must **not** use the
  deprecated legacy `cusparseScsrmv`.

Run `python grade.py --check-solution` to grade the reference solution instead.
