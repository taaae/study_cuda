# Exercise 20 — Conjugate Gradient (capstone)

**New concepts:** assembling a real numerical algorithm — the **Conjugate
Gradient (CG)** iterative solver for a symmetric positive-definite (SPD) sparse
system `A x = b` — out of the GPU primitives you already built: **SpMV**,
**dot products** (a reduction), and **AXPY** (`y += a*x`).

This is the payoff. Each CG step is just a few of those primitives chained
together, with the scalars (`alpha`, `beta`, residual norm) computed from
on-device dot products. Nothing new about CUDA here — the lesson is *composition*.

## The math, one step at a time

CG solves `A x = b` for SPD `A` by building a sequence of search directions `p`
that are mutually *A-conjugate*. Starting from `x0 = 0`:

```
r = b - A*x0 = b          (initial residual; x0 = 0 so r = b)
p = r                     (initial search direction)
rsold = r·r               (squared residual norm)

repeat (up to maxiter):
    Ap    = A * p                       # SpMV
    alpha = rsold / (p · Ap)            # dot product, then scalar divide
    x     = x + alpha * p               # AXPY
    r     = r - alpha * Ap              # AXPY
    rsnew = r · r                       # dot product
    if sqrt(rsnew) / sqrt(b·b) < tol: stop
    beta  = rsnew / rsold              # scalar
    p     = r + beta * p               # scaled AXPY (xpay): p = r + beta*p
    rsold = rsnew
```

| CG step | GPU primitive |
|---------|---------------|
| `Ap = A*p` | **SpMV** (your CSR kernel) |
| `p·Ap`, `r·r`, `b·b` | **dot product** = reduction, kept on-device |
| `x += alpha*p`, `r -= alpha*Ap` | **AXPY** |
| `p = r + beta*p` | **XPAY** (scale the destination, then add) |

Keep the dot products **on the device** — copy back only the single scalar you
need (`p·Ap`, `rsnew`) to compute `alpha`/`beta` on the host. Copying whole
vectors every iteration would dominate the runtime.

## The task

Implement CG in `cg.cu`. You provide the kernels and the host loop; the harness
provides `A` (an SPD 2-D 5-point Laplacian), a right-hand side `b`, an `x`
pre-zeroed for you, and the convergence parameters. Write the solution into `x`.

Suggested kernels (all simple, all things you've built before):

- `spmv_csr` — `Ap = A*p` (CSR, one thread per row is fine; A is well-balanced).
- `dot` — `out += sum(a[i]*b[i])` via a block reduction + `atomicAdd` to a single
  device scalar (zero it before each call).
- `axpy` — `y[i] += a * x[i]`.
- `xpay` — `p[i] = x[i] + a * p[i]` (note: scales the *destination*).

### `solve` signature (the contract)

```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* b, float* x, int n, int maxiter, float tol);
```

All pointers are **device pointers**. `A` is CSR (`rowPtr[n+1]`, `colIdx[nnz]`,
`vals[nnz]`), `n × n`, SPD. `x` is **already zeroed** by the harness — `solve`
writes the solution. Iterate until `||b - A x|| / ||b|| < tol` or `maxiter`
iterations. You allocate your own scratch vectors (`r`, `p`, `Ap`) and the device
scalar inside `solve`, and free them before returning.

## Syntax / reference

- A block reduction into a single device accumulator:
  ```cpp
  __global__ void dot(const float* a, const float* b, float* out, int n) {
      __shared__ float s[256];
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      float v = (i < n) ? a[i] * b[i] : 0.0f;
      s[threadIdx.x] = v; __syncthreads();
      for (int off = blockDim.x / 2; off > 0; off >>= 1) {
          if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
          __syncthreads();
      }
      if (threadIdx.x == 0) atomicAdd(out, s[0]);   // zero *out before launch
  }
  ```
  (Warp-shuffle reductions from earlier exercises also work; this is the simplest
  correct version.)
- Pull a scalar back with `cudaMemcpy(&host, d_scalar, sizeof(float),
  cudaMemcpyDeviceToHost)`.
- Initialization shortcut: because `x0 = 0`, `r = b - A*x0 = b`, so you can just
  copy `b` into `r` and `p` (`cudaMemcpy` device-to-device) — no SpMV needed for
  the initial residual.
- Compute `alpha`/`beta` on the host from the scalars you copied back; pass them
  into the `axpy`/`xpay` kernels by value.

## Grading (`!python grade.py`)

- **correctness** — the harness builds `b = A x_true` for a known `x_true`, runs
  your `solve`, then recomputes the **relative residual** `||b - A x|| / ||b||`
  and requires `rel_resid <= 1e-3`.
- **metric** — reports `rel_resid` and total `ms`.
- **source** — must contain a CSR SpMV-style loop (uses `rowPtr`) and a
  dot/reduction.

Run `python grade.py --check-solution` to grade the reference solution instead.
