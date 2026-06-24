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
    // TODO: __host__ __device__ bool operator()(float x) const { return x > t; }
};

// Unary op: square a value (used by transform_reduce).
struct square {
    // TODO: __host__ __device__ float operator()(float x) const { return x * x; }
};

// All pointers are DEVICE pointers. out_count and out_sumsq are single values.
void solve(const float* in, int n, float threshold,
           float* out_compacted, int* out_count, float* out_sumsq) {
    // Wrap raw device pointers so Thrust can iterate over them.
    // TODO: thrust::device_ptr<const float> in_ptr(in);
    // TODO: thrust::device_ptr<float>       out_ptr(out_compacted);

    // (a) Stream-compact elements > threshold into out_compacted.
    // The returned iterator minus out_ptr is the number of kept elements.
    // TODO: auto end = thrust::copy_if(thrust::device, in_ptr, in_ptr + n,
    //                                  out_ptr, keep_above{threshold});
    // TODO: int count = (int)(end - out_ptr);

    // (b) Sum of squares of the KEPT elements (out_ptr .. out_ptr + count).
    // TODO: float ss = thrust::transform_reduce(thrust::device,
    //                       out_ptr, out_ptr + count,
    //                       square{}, 0.0f, thrust::plus<float>());

    // Write the two scalar results back to their single-value device outputs.
    // TODO: CUDA_CHECK(cudaMemcpy(out_count, &count, sizeof(int),   cudaMemcpyHostToDevice));
    // TODO: CUDA_CHECK(cudaMemcpy(out_sumsq, &ss,    sizeof(float), cudaMemcpyHostToDevice));
}
