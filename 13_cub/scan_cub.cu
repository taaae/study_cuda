// Exercise 13 — Exclusive scan with CUB's DeviceScan.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// CUB's device-wide primitives.
#include <cub/cub.h>

// in/out are DEVICE pointers of length n. Use the two-call temp-storage idiom.
void solve(const int* in, int* out, int n) {
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;

    // 1) Sizing pass: call cub::DeviceScan::ExclusiveSum with d_temp == nullptr
    //    so CUB just fills temp_bytes.
    // TODO: sizing call.

    // 2) Allocate the scratch CUB asked for.
    // TODO: cudaMalloc d_temp to temp_bytes.

    // 3) Real pass: same ExclusiveSum call, now with valid scratch; then free it.
    // TODO: run the scan and free the scratch. (See README + hints.md.)
}
