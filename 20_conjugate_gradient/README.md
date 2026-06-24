# Exercise 20 — Conjugate Gradient (capstone)
> A real numerical solver, built entirely from SpMV + dot + axpy — the primitives you already have.

## The idea

This is the payoff for the whole sparse track. The **Conjugate Gradient (CG)**
method solves a linear system `A x = b` when `A` is **symmetric positive-definite
(SPD)** — the kind of matrix that comes out of discretizing physics (heat
diffusion, electrostatics, structural mechanics) and out of the normal equations
in least-squares and many ML problems. CG is *iterative*: instead of factoring `A`
(impossible at scale for a sparse matrix), it nudges a guess `x` toward the answer,
one cheap step at a time.

The beautiful part for us: **each CG step is just a handful of GPU primitives you've
already written** — one SpMV, two dot products, two AXPYs, one scaled-add. There's
nothing new about CUDA here. The lesson is **composition**: how a serious algorithm
falls out of small, well-understood kernels.

## The algorithm, one step at a time

CG builds a sequence of search directions `p` that are mutually *A-conjugate*
(`pᵢᵀ A pⱼ = 0` for `i ≠ j`), which is what lets it converge so fast. Starting from
`x₀ = 0`:

```text
r = b - A*x0 = b          # initial residual; x0 = 0 so r = b (no SpMV needed!)
p = r                     # initial search direction
rsold = r·r               # squared residual norm
bnorm = sqrt(b·b)         # for the relative-residual test

repeat (up to maxiter):
    Ap    = A * p                       # 1) SpMV
    alpha = rsold / (p · Ap)            # 2) dot, then a scalar divide on the host
    x     = x + alpha * p               # 3) AXPY  (step toward the solution)
    r     = r - alpha * Ap              # 4) AXPY  (note the MINUS)
    rsnew = r · r                       # 5) dot   (new residual norm^2)
    if sqrt(rsnew) / bnorm < tol: stop  #    converged?
    beta  = rsnew / rsold               #    scalar
    p     = r + beta * p                # 6) XPAY  (scales the DESTINATION p)
    rsold = rsnew
```

Each line maps to exactly one primitive:

| CG step | GPU primitive |
|---------|---------------|
| `Ap = A*p` | **SpMV** — your CSR kernel |
| `p·Ap`, `r·r`, `b·b` | **dot product** = a reduction, kept on-device |
| `x += alpha*p`, `r -= alpha*Ap` | **AXPY** — `y[i] += a*x[i]` |
| `p = r + beta*p` | **XPAY** — scale the destination, then add (`p[i] = x[i] + a*p[i]`) |

> **Why CG converges fast:** in exact arithmetic CG reaches the true solution of an
> `n×n` system in at most `n` steps, but in practice the residual drops *much*
> sooner — governed by the spread of `A`'s eigenvalues (its condition number). The
> harness's 2-D Laplacian on a 256×256 grid (`n = 65536`) converges in a few
> hundred iterations, nowhere near 65536. Real solvers add a *preconditioner* to
> shrink the condition number and cut the iteration count further.

## Under the hood: keep the scalars on the device

A naive port copies whole vectors back to the host each iteration — that data
movement would dwarf the actual math and crush performance. The trick is to keep
every vector resident on the GPU and copy back **only the single scalar** you need
to compute `alpha` and `beta`:

- A dot product is a reduction into one `float`. Each call: `cudaMemset` your device
  scalar to 0, launch the `dot` kernel (block reduction + `atomicAdd` into that
  scalar), then `cudaMemcpy` *just that one float* back.
- Compute `alpha = rsold/pAp` and `beta = rsnew/rsold` on the host, then pass them
  **by value** into the `axpy`/`xpay` kernels.

So the only host↔device traffic per iteration is a couple of 4-byte scalars. The
heavy data never leaves the GPU.

## A picture

```text
One CG iteration — data flows GPU-resident; only scalars cross to the host:

      p ──► [ SpMV: Ap = A*p ] ──► Ap
                                    │
      p, Ap ──► [ dot ] ─► pAp ─(copy 4B)─► host: alpha = rsold / pAp
                                                        │ (by value)
              ┌───────────────────────────────────────┘
              ▼
   [ axpy:  x += alpha*p ]        (x advances toward solution)
   [ axpy:  r -= alpha*Ap ]       (residual shrinks; note the minus)
              │
      r,r ──► [ dot ] ─► rsnew ─(copy 4B)─► host: converged? beta = rsnew/rsold
              │                                          │ (by value)
              ▼                                          ▼
   [ xpay:  p = r + beta*p ]   <───────────────────────┘   then rsold = rsnew
```

## Your task

Implement CG in `cg.cu` — both the kernels and the host loop. The harness provides
`A` (an SPD 2-D 5-point Laplacian in CSR), a right-hand side `b`, an `x` that's
**already zeroed**, and the convergence parameters. Write the solution into `x`.

Suggested kernels (all things you've built before):

- `spmv_csr` — `Ap = A*p` (CSR, one thread per row is fine; this `A` is well-balanced).
- `dot` — block reduction + `atomicAdd` into a single device scalar (zero it first).
- `axpy` — `y[i] += a * x[i]`.
- `xpay` — `p[i] = x[i] + a * p[i]` (scales the **destination**).

### The `solve` contract

```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* b, float* x, int n, int maxiter, float tol);
```

All pointers are **device pointers**. `A` is CSR (`rowPtr[n+1]`, `colIdx[nnz]`,
`vals[nnz]`), `n×n`, SPD. `x` is **already zeroed** by the harness — `solve` writes
the solution. Iterate until `||b - A x|| / ||b|| < tol` or `maxiter` iterations.
You allocate your own scratch vectors (`r`, `p`, `Ap`) and the device scalar inside
`solve`, **and free them before returning** — the harness calls `solve` many times
when timing, so a leak will exhaust GPU memory.

## Functions & syntax you'll need

| Tool | Form | Role |
|------|------|------|
| SpMV row | `int row = blockIdx.x*blockDim.x + threadIdx.x;` then loop `rowPtr[row]..rowPtr[row+1]` | `Ap = A*p` |
| `__shared__` | `__shared__ float s[256];` | per-block scratch for the dot reduction |
| `__syncthreads()` | barrier inside a block | between reduction steps |
| `atomicAdd` | `atomicAdd(out, s[0])` | combine each block's partial into the one device scalar (zero `*out` first!) |
| `cudaMemset` | `cudaMemset(d_scalar, 0, sizeof(float))` | zero the dot accumulator before **each** dot |
| `cudaMemcpy` (D2D) | `cudaMemcpy(r, b, n*sizeof(float), cudaMemcpyDeviceToDevice)` | init `r = b`, `p = b` with no SpMV |
| `cudaMemcpy` (D2H) | `cudaMemcpy(&host, d_scalar, sizeof(float), cudaMemcpyDeviceToHost)` | pull a scalar back |
| `cudaMalloc` / `cudaFree` | scratch `r`, `p`, `Ap`, `d_scalar` | allocate / free per `solve` call |
| `ceil_div`, `sqrtf` | from `cuda_utils.cuh` / `<cmath>` | grid sizing / residual norms |

The simplest correct `dot` (warp-shuffle versions from earlier exercises also work):

```cpp
__global__ void dot(const float* a, const float* b, float* out, int n) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (i < n) ? a[i] * b[i] : 0.0f;
    __syncthreads();
    for (int off = blockDim.x/2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);   // zero *out before launch
}
```

**Init shortcut:** because `x₀ = 0`, `r₀ = b - A*x₀ = b`. Just `cudaMemcpy` `b`
into both `r` and `p` (device-to-device) — no SpMV for the initial residual.

## How it's graded

Run `python grade.py` (or `--check-solution` for the reference).

- **correctness** — the harness builds `b = A·x_true` for a known `x_true`, runs
  your `solve`, then recomputes the **relative residual** `||b - A x|| / ||b||` and
  requires `rel_resid ≤ 1e-3`. (The harness solves to `tol = 1e-4`, harder than the
  grading bar, so a correct CG comfortably passes.)
- **metric** — reports `rel_resid` and total `ms`.
- **source** — must contain a CSR SpMV-style loop (uses `rowPtr`) and an on-device
  reduction (`atomicAdd` / `__shfl_down_sync` / `__syncthreads`).

## Going deeper — easy ways to get a wrong answer

- **Forgetting to `cudaMemset` the dot scalar to 0 before *each* dot.** Stale sums
  accumulate and blow up `alpha`/`beta`.
- **Sign error:** `x` *increases* by `alpha*p`, but `r` *decreases* by `alpha*Ap`
  — pass `-alpha` to that AXPY.
- **`xpay` scales the destination:** `p = r + beta*p`, not `p = beta*r + p`. The
  latter is a different (wrong) recurrence.
- **Update order:** set `rsold = rsnew` only *after* computing `beta`.
- **Leaking scratch:** `cudaFree` everything before returning, or the timed reruns
  run the GPU out of memory.

Beyond correctness: this exact SpMV+dot+axpy skeleton is also how **BiCGSTAB**,
**GMRES**, and Krylov eigensolvers are built — swap the recurrence, keep the
primitives. And the single biggest real-world speedup is **preconditioning**
(e.g. Jacobi / incomplete-Cholesky): solve `M⁻¹A x = M⁻¹b` with `M ≈ A`, slashing
the iteration count. You now have every building block to write one.
