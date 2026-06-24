// Reference solution — Exercise 19 (cuSPARSE generic SpMV).
#include "cuda_utils.cuh"
#include <cusparse.h>

#define CUSPARSE_CHECK(call)                                                    \
  do {                                                                          \
    cusparseStatus_t _s = (call);                                              \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuSPARSE error %s:%d: '%s' -> %s\n", __FILE__,      \
                   __LINE__, #call, cusparseGetErrorString(_s));               \
      std::exit(1);                                                            \
    }                                                                           \
  } while (0)

void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows, int ncols, int nnz) {
    float alpha = 1.0f, beta = 0.0f;

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Sparse matrix A in CSR. The descriptor stores const-correct pointers as
    // void*; cuSPARSE does not modify the CSR arrays for SpMV.
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA, nrows, ncols, nnz,
        (void*)rowPtr, (void*)colIdx, (void*)vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, ncols, (void*)x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, nrows, (void*)y, CUDA_R_32F));

    // Buffer pattern: query size, allocate, then run.
    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    void* dBuffer = nullptr;
    if (bufferSize > 0) CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    CUSPARSE_CHECK(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    if (dBuffer) CUDA_CHECK(cudaFree(dBuffer));
    CUSPARSE_CHECK(cusparseDestroy(handle));
}
