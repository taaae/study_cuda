// Exercise 20 — Conjugate Gradient (capstone).
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// ---- Kernels you assemble CG from -----------------------------------------

// Ap = A * p   (CSR, one thread per row).
__global__ void spmv_csr(const int* rowPtr, const int* colIdx, const float* vals,
                         const float* p, float* Ap, int n) {
    // TODO: one thread per row; sum vals[j]*p[colIdx[j]] over rowPtr[row]..rowPtr[row+1].
}

// out += sum_i a[i]*b[i].  Caller MUST zero *out before launching.
__global__ void dot(const float* a, const float* b, float* out, int n) {
    // TODO: per-thread product -> shared-memory block reduction -> atomicAdd(out, s[0]).
}

// y[i] += alpha * x[i]
__global__ void axpy(float alpha, const float* x, float* y, int n) {
    // TODO: y[i] += alpha * x[i] with a boundary guard.
}

// p[i] = x[i] + beta * p[i]   (scales the DESTINATION, then adds x)
__global__ void xpay(const float* x, float beta, float* p, int n) {
    // TODO: p[i] = x[i] + beta * p[i] with a boundary guard.
}

// ---- Host driver ----------------------------------------------------------
// A is CSR (rowPtr[n+1], colIdx[nnz], vals[nnz]), n x n, SPD. x is pre-zeroed.
// Solve A x = b to ||b - A x|| / ||b|| < tol, or maxiter iterations.
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* b, float* x, int n, int maxiter, float tol) {
    // TODO: allocate device scratch r, p, Ap (length n) and a device scalar.
    // TODO: x is already 0, so r = b and p = b (device-to-device copies).
    // TODO: rsold = r·r ;  bnorm2 = b·b   (dot kernel, scalar copied back).
    // TODO: loop:
    //   Ap = A*p (spmv_csr)
    //   pAp = p·Ap (dot)
    //   alpha = rsold / pAp
    //   x += alpha*p  ;  r -= alpha*Ap   (axpy, note the minus for r)
    //   rsnew = r·r (dot)
    //   if sqrt(rsnew)/sqrt(bnorm2) < tol: break
    //   beta = rsnew / rsold
    //   p = r + beta*p   (xpay)
    //   rsold = rsnew
    // TODO: free your scratch before returning.
}
