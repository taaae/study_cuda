// Exercise 13 — Exclusive scan with CUB's DeviceScan.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// CUB's device-wide primitives.
#include <cub/cub.h>

// in/out are DEVICE pointers of length n. Use the two-call temp-storage idiom.
void solve(const int* in, int* out, int n) {
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;

    // 1) Sizing pass: with d_temp == nullptr, CUB just fills temp_bytes.
    // TODO: cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);

    // 2) Allocate the scratch CUB asked for.
    // TODO: CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    // 3) Real pass: same call, now with valid scratch.
    // TODO: cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);

    // TODO: CUDA_CHECK(cudaFree(d_temp));
}
