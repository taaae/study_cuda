// Exercise 12 — Thrust: stream compaction + fused transform-reduce.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Thrust headers you'll need.
#include <thrust/copy.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

// Predicate: keep elements strictly greater than a threshold.
struct keep_above {
    float t;
    // TODO: add a __host__ __device__ operator() returning whether x > t.
};

// Unary op: square a value (used by transform_reduce).
struct square {
    // TODO: add a __host__ __device__ operator() returning x squared.
};

// All pointers are DEVICE pointers. out_count and out_sumsq are single values.
void solve(const float* in, int n, float threshold,
           float* out_compacted, int* out_count, float* out_sumsq) {
    // TODO: wrap the raw device pointers in thrust::device_ptr so Thrust can
    //       iterate over them.

    // (a) Stream-compact elements > threshold into out_compacted.
    // TODO: use thrust::copy_if with the keep_above predicate; the count is the
    //       returned iterator minus the output begin.

    // (b) Sum of squares of the KEPT elements only.
    // TODO: use thrust::transform_reduce over the kept range with the square op
    //       and plus to get the sum of squares.

    // TODO: copy the two scalar results back to their single-value device
    //       outputs. (See README + hints.md.)
}
