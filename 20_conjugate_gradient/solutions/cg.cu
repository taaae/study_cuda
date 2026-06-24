// Reference solution — Exercise 20 (Conjugate Gradient capstone).
#include "cuda_utils.cuh"
#include <cmath>

#define CG_BLOCK 256

// Ap = A * p   (CSR, one thread per row).
__global__ void spmv_csr(const int* rowPtr, const int* colIdx, const float* vals,
                         const float* p, float* Ap, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float sum = 0.0f;
    int start = rowPtr[row], end = rowPtr[row + 1];
    for (int j = start; j < end; ++j) sum += vals[j] * p[colIdx[j]];
    Ap[row] = sum;
}

// out += sum_i a[i]*b[i].  Caller must zero *out first.
__global__ void dot(const float* a, const float* b, float* out, int n) {
    __shared__ float s[CG_BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] * b[i] : 0.0f;
    s[threadIdx.x] = v;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

// y[i] += alpha * x[i]
__global__ void axpy(float alpha, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

// p[i] = x[i] + beta * p[i]
__global__ void xpay(const float* x, float beta, float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = x[i] + beta * p[i];
}

// Host helper: out = a·b, returned to host. d_scalar is reused device scratch.
static float device_dot(const float* a, const float* b, float* d_scalar,
                        int n, int grid) {
    CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(float)));
    dot<<<grid, CG_BLOCK>>>(a, b, d_scalar, n);
    float h = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h, d_scalar, sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* b, float* x, int n, int maxiter, float tol) {
    int grid = ceil_div(n, CG_BLOCK);

    float *r, *p, *Ap, *d_scalar;
    CUDA_CHECK(cudaMalloc(&r,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Ap, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scalar, sizeof(float)));

    // x0 = 0  =>  r0 = b - A*x0 = b,  p0 = r0 = b.
    CUDA_CHECK(cudaMemcpy(r, b, n * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(p, b, n * sizeof(float), cudaMemcpyDeviceToDevice));

    float bnorm2 = device_dot(b, b, d_scalar, n, grid);
    float rsold  = device_dot(r, r, d_scalar, n, grid);
    float bnorm  = std::sqrt(bnorm2);
    if (bnorm == 0.0f) bnorm = 1.0f;  // b == 0 -> x == 0 already correct

    for (int it = 0; it < maxiter; ++it) {
        // Ap = A * p
        spmv_csr<<<grid, CG_BLOCK>>>(rowPtr, colIdx, vals, p, Ap, n);

        // alpha = rsold / (p·Ap)
        float pAp = device_dot(p, Ap, d_scalar, n, grid);
        if (pAp == 0.0f) break;
        float alpha = rsold / pAp;

        // x += alpha*p ;  r -= alpha*Ap
        axpy<<<grid, CG_BLOCK>>>( alpha, p,  x, n);
        axpy<<<grid, CG_BLOCK>>>(-alpha, Ap, r, n);

        // convergence test on the recurrence residual
        float rsnew = device_dot(r, r, d_scalar, n, grid);
        if (std::sqrt(rsnew) / bnorm < tol) break;

        float beta = rsnew / rsold;
        // p = r + beta*p
        xpay<<<grid, CG_BLOCK>>>(r, beta, p, n);
        rsold = rsnew;
    }
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaFree(r));
    CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(Ap));
    CUDA_CHECK(cudaFree(d_scalar));
}
