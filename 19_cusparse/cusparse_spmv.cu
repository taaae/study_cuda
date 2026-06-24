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

    // Use the cuSPARSE generic API. The overall flow:
    //   - create a handle, a CSR descriptor for A, and dense-vector descriptors
    //     for x and y (32-bit indices, base 0, CUDA_R_32F);
    //   - query the SpMV buffer size, allocate that scratch buffer, run the SpMV
    //     (non-transpose, ALG_DEFAULT), then tear everything down.
    // TODO: implement the steps above. (See README's function table + hints.md.)
}
