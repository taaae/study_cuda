// Exercise 20 — Conjugate Gradient (capstone).
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// ---- Kernels you assemble CG from -----------------------------------------

// Ap = A * p   (CSR, one thread per row).
__global__ void spmv_csr(const int* rowPtr, const int* colIdx, const float* vals,
                         const float* p, float* Ap, int n) {
    // TODO: one thread per row; accumulate this row's CSR nonzeros into Ap.
}

// out += sum_i a[i]*b[i].  Caller MUST zero *out before launching.
__global__ void dot(const float* a, const float* b, float* out, int n) {
    // TODO: per-thread products, block reduce in shared memory, atomicAdd into *out.
}

// y[i] += alpha * x[i]
__global__ void axpy(float alpha, const float* x, float* y, int n) {
    // TODO: scaled vector add y += alpha*x, one guarded thread per element.
}

// p[i] = x[i] + beta * p[i]   (scales the DESTINATION, then adds x)
__global__ void xpay(const float* x, float beta, float* p, int n) {
    // TODO: scale the destination then add x (p = x + beta*p), one guarded
    //       thread per element.
}

// ---- Host driver ----------------------------------------------------------
// A is CSR (rowPtr[n+1], colIdx[nnz], vals[nnz]), n x n, SPD. x is pre-zeroed.
// Solve A x = b to ||b - A x|| / ||b|| < tol, or maxiter iterations.
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* b, float* x, int n, int maxiter, float tol) {
    // Assemble standard CG from the kernels above.
    // TODO: allocate device scratch (r, p, Ap of length n) plus a device scalar
    //       to receive dot results.
    // TODO: initialize the residual and search direction from the fact that x=0,
    //       and seed rsold = r·r and the b-norm used for the stopping test.
    // TODO: each iteration is one SpMV + two dot products + axpy/xpay updates:
    //       form Ap, get the step size from rsold and p·Ap, update x and the
    //       residual r (mind the sign), recompute the residual norm, test for
    //       convergence, then build the next direction from beta = rsnew/rsold.
    //       (See README's function table + hints.md for the exact recurrences.)
    // TODO: free your scratch before returning.
}
