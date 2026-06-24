// Reference solution — Exercise 12. Thrust stream compaction + transform-reduce.
#include "cuda_utils.cuh"

#include <thrust/copy.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

struct keep_above {
    float t;
    __host__ __device__ bool operator()(float x) const { return x > t; }
};

struct square {
    __host__ __device__ float operator()(float x) const { return x * x; }
};

void solve(const float* in, int n, float threshold,
           float* out_compacted, int* out_count, float* out_sumsq) {
    thrust::device_ptr<const float> in_ptr(in);
    thrust::device_ptr<float>       out_ptr(out_compacted);

    // (a) stream compaction
    auto end = thrust::copy_if(thrust::device, in_ptr, in_ptr + n,
                               out_ptr, keep_above{threshold});
    int count = (int)(end - out_ptr);

    // (b) fused sum of squares over the kept elements
    float ss = thrust::transform_reduce(thrust::device,
                                        out_ptr, out_ptr + count,
                                        square{}, 0.0f, thrust::plus<float>());

    CUDA_CHECK(cudaMemcpy(out_count, &count, sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_sumsq, &ss,    sizeof(float), cudaMemcpyHostToDevice));
}
