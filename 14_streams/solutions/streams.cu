// Reference solution — Exercise 14.
#include <vector>
#include <algorithm>
#include "cuda_utils.cuh"

__global__ void map_kernel(const float* x, float* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        float v = x[i];
        y[i] = sqrtf(v) * v + 1.0f;
    }
}

void solve(const float* h_in, float* h_out, int n, int nStreams) {
    const int block = 256;
    int chunk = ceil_div(n, nStreams);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));

    std::vector<cudaStream_t> streams(nStreams);
    for (int s = 0; s < nStreams; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    for (int off = 0, i = 0; off < n; off += chunk, ++i) {
        int len = std::min(chunk, n - off);
        size_t b = (size_t)len * sizeof(float);
        cudaStream_t s = streams[i % nStreams];
        CUDA_CHECK(cudaMemcpyAsync(d_in + off, h_in + off, b,
                                   cudaMemcpyHostToDevice, s));
        int grid = ceil_div(len, block);
        map_kernel<<<grid, block, 0, s>>>(d_in + off, d_out + off, len);
        CUDA_CHECK(cudaMemcpyAsync(h_out + off, d_out + off, b,
                                   cudaMemcpyDeviceToHost, s));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    for (auto s : streams) CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}
