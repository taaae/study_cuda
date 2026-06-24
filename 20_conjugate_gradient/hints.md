# Hints — Exercise 20 (Conjugate Gradient capstone)

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — The skeleton of one CG iteration (no code)</summary>

Each iteration is exactly: one **SpMV** (`Ap = A*p`), one **dot** (`p·Ap`) to get
`alpha`, two **AXPYs** (`x += alpha*p`, `r -= alpha*Ap`), one more **dot** (`r·r`)
to test convergence and form `beta`, and one **XPAY** (`p = r + beta*p`). That's
the whole loop. Every piece is a kernel you've already written.
</details>

<details>
<summary>Hint 2 — Initialization without an extra SpMV (concept)</summary>

The harness zeroes `x` for you. With `x0 = 0`, the initial residual
`r0 = b - A*x0 = b`. So just copy `b` into both `r` and `p` (device-to-device
`cudaMemcpy`). Compute `rsold = r·r` and `bnorm2 = b·b` once up front; you'll
divide by `sqrt(bnorm2)` for the relative-residual test.
</details>

<details>
<summary>Hint 3 — Keeping dot products on the device (concept)</summary>

A dot product is a reduction. Each call: `cudaMemset` your single device scalar to
0, launch the `dot` kernel (block reduction + `atomicAdd` into that scalar), then
`cudaMemcpy` just the one float back. Never copy whole vectors per iteration — the
only host↔device traffic per step is a couple of scalars.
</details>

<details>
<summary>Hint 4 — The kernels (code)</summary>

```cpp
__global__ void spmv_csr(const int* rowPtr, const int* colIdx, const float* vals,
                         const float* p, float* Ap, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float s = 0.0f;
    for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) s += vals[j] * p[colIdx[j]];
    Ap[row] = s;
}
__global__ void dot(const float* a, const float* b, float* out, int n) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (i < n) ? a[i]*b[i] : 0.0f; __syncthreads();
    for (int o = blockDim.x/2; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);
}
__global__ void axpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) y[i] += a*x[i];
}
__global__ void xpay(const float* x, float b, float* p, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) p[i] = x[i] + b*p[i];
}
```
</details>

<details>
<summary>Hint 5 — The host loop (code)</summary>

```cpp
// r = b; p = b; rsold = r·r; bnorm = sqrt(b·b)
for (int it = 0; it < maxiter; ++it) {
    spmv_csr<<<grid,256>>>(rowPtr, colIdx, vals, p, Ap, n);
    float pAp   = device_dot(p, Ap);          // zero scalar, launch dot, copy back
    float alpha = rsold / pAp;
    axpy<<<grid,256>>>( alpha, p,  x, n);
    axpy<<<grid,256>>>(-alpha, Ap, r, n);     // note the MINUS
    float rsnew = device_dot(r, r);
    if (sqrtf(rsnew) / bnorm < tol) break;
    float beta = rsnew / rsold;
    xpay<<<grid,256>>>(r, beta, p, n);        // p = r + beta*p
    rsold = rsnew;
}
```
</details>

<details>
<summary>Hint 6 — Easy ways to get a wrong answer (concept)</summary>

- Forgetting to `cudaMemset` the dot scalar to 0 before *each* dot — stale sums
  blow up `alpha`/`beta`.
- Sign error: `r` decreases by `alpha*Ap` (use `-alpha`), while `x` increases by
  `alpha*p`.
- `xpay` scales the **destination** `p`, not `x`: `p = r + beta*p`. Writing
  `p = beta*r + p` is a different (wrong) recurrence.
- Updating `rsold = rsnew` at the *end* of the iteration, after computing `beta`.
- Don't forget to `cudaFree` your scratch (`r`, `p`, `Ap`, scalar) — the harness
  calls `solve` many times when timing, so a leak will exhaust GPU memory.
</details>
