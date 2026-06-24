// Exercise 19 — cuSPARSE generic SpMV.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"
#include <cusparse.h>

// Status check for cuSPARSE calls, analogous to CUDA_CHECK. (Non-learning
// scaffolding — provided so you can focus on the API sequence.)
#define CUSPARSE_CHECK(call)                                                    \
  do {                                                                          \
    cusparseStatus_t _s = (call);                                              \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuSPARSE error %s:%d: '%s' -> %s\n", __FILE__,      \
                   __LINE__, #call, cusparseGetErrorString(_s));               \
      std::exit(1);                                                            \
    }                                                                           \
  } while (0)

// Compute y = alpha*A*x + beta*y for CSR A using the MODERN GENERIC API.
// All pointers are DEVICE pointers. CSR: rowPtr[nrows+1], colIdx[nnz], vals[nnz].
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows, int ncols, int nnz) {
    float alpha = 1.0f, beta = 0.0f;   // plain y = A*x; passed BY POINTER below

    // TODO: create the cuSPARSE handle (cusparseCreate).

    // TODO: create the sparse-matrix descriptor for A with cusparseCreateCsr.
    //       Indices are 32-bit (CUSPARSE_INDEX_32I), base 0
    //       (CUSPARSE_INDEX_BASE_ZERO), values CUDA_R_32F.

    // TODO: create dense-vector descriptors for x (length ncols) and y
    //       (length nrows) with cusparseCreateDnVec, type CUDA_R_32F.

    // TODO: query the required buffer size with cusparseSpMV_bufferSize using
    //       CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha/&beta, CUDA_R_32F,
    //       CUSPARSE_SPMV_ALG_DEFAULT. Then cudaMalloc the external buffer.

    // TODO: run cusparseSpMV with the SAME args plus the buffer pointer.

    // TODO: tear down — destroy the matrix/vector descriptors, free the buffer,
    //       destroy the handle.
}
