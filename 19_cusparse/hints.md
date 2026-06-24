# Hints — Exercise 19 (cuSPARSE generic SpMV)

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — The shape of a generic-API call (no code)</summary>

Library SpMV is four phases: (1) make a **handle** (the library context), (2)
**describe** your data as opaque descriptors — one for the sparse matrix, one per
dense vector, (3) **ask** how much scratch the op needs, allocate it, and **run**,
(4) **destroy** everything you created. You never write a kernel.
</details>

<details>
<summary>Hint 2 — Describing CSR and the vectors (concept)</summary>

`cusparseCreateCsr` takes your three CSR device pointers plus dims and *type tags*
that tell the library how to interpret the bytes: the index arrays are 32-bit
integers (`CUSPARSE_INDEX_32I`), the matrix is 0-based (`CUSPARSE_INDEX_BASE_ZERO`),
the values are `float` (`CUDA_R_32F`). `cusparseCreateDnVec` is simpler: length +
device pointer + value type. `x` has length `ncols`, `y` has length `nrows`.
</details>

<details>
<summary>Hint 3 — The buffer pattern (concept)</summary>

Don't guess the scratch size. Call `cusparseSpMV_bufferSize` first with *exactly*
the arguments you'll pass to `cusparseSpMV` (same `alpha`/`beta`, descriptors,
operation, compute type, algorithm). It writes a `size_t`. `cudaMalloc` that many
bytes (it can legitimately be 0), then pass the pointer to `cusparseSpMV`.
</details>

<details>
<summary>Hint 4 — alpha/beta and the operation (concept)</summary>

`alpha` and `beta` are passed **by pointer**, not by value — here they're host
floats, so pass `&alpha` / `&beta`. Use `CUSPARSE_OPERATION_NON_TRANSPOSE` (plain
`A*x`, not `A^T x`) and `CUSPARSE_SPMV_ALG_DEFAULT`. The compute type `CUDA_R_32F`
must match your float data.
</details>

<details>
<summary>Hint 5 — The descriptors + buffer (code)</summary>

```cpp
cusparseHandle_t handle;
CUSPARSE_CHECK(cusparseCreate(&handle));

cusparseSpMatDescr_t matA;
CUSPARSE_CHECK(cusparseCreateCsr(&matA, nrows, ncols, nnz,
    (void*)rowPtr, (void*)colIdx, (void*)vals,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

cusparseDnVecDescr_t vecX, vecY;
CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, ncols, (void*)x, CUDA_R_32F));
CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, nrows, (void*)y, CUDA_R_32F));

size_t bufferSize = 0;
CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
void* dBuffer = nullptr;
if (bufferSize > 0) CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
```
</details>

<details>
<summary>Hint 6 — Run and tear down (code)</summary>

```cpp
CUSPARSE_CHECK(cusparseSpMV(handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

CUSPARSE_CHECK(cusparseDestroySpMat(matA));
CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
if (dBuffer) CUDA_CHECK(cudaFree(dBuffer));
CUSPARSE_CHECK(cusparseDestroy(handle));
```

Don't call the legacy `cusparseScsrmv` — it's deprecated and the grader rejects it.
</details>
