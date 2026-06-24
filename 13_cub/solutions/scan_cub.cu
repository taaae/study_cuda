// Reference solution — Exercise 13. Exclusive scan via cub::DeviceScan.
#include "cuda_utils.cuh"

#include <cub/cub.h>

void solve(const int* in, int* out, int n) {
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;

    // Sizing pass.
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);

    // Allocate scratch and run for real.
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);

    CUDA_CHECK(cudaFree(d_temp));
}
