#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: runs on the GPU, once per thread
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    // Each thread computes one element
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    pritnf("HIIIII\n");
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    // --- Allocate host memory ---
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Fill input arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // --- Allocate device (GPU) memory ---
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // --- Copy data from host to device ---
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // --- Launch kernel ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // --- Copy result back to host ---
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // --- Verify result ---
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Mismatch at index %d: got %f, expected %f\n",
                   i, h_c[i], h_a[i] + h_b[i]);
            ok = false;
            break;
        }
    }
    if (ok) printf("All %d results correct! c[0]=%.0f, c[512]=%.0f, c[1023]=%.0f\n",
                   N, h_c[0], h_c[512], h_c[1023]);

    // --- Free memory ---
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a);    free(h_b);    free(h_c);

    return 0;
}