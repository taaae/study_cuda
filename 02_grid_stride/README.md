# Exercise 02 — Grid-Stride Loops
> Stop sizing your grid to the data. Launch a fixed pool of threads and let each one walk the array — one launch config that works for any `n`.

## The idea

In exercise 01 you launched *one thread per element*. That works, but it ties your launch configuration to the problem size: 16M elements meant ~65,000 blocks. What if `n` is a billion? What if it changes every call? You'd recompute the grid every time, and you might even exceed the maximum grid dimension.

The **grid-stride loop** flips the relationship. You launch a **fixed, modest** number of threads — sized to the *machine*, not the data — and each thread processes *many* elements by striding across the array in jumps. The jump size is the total number of threads in the grid, so the threads collectively tile the whole array with no gaps and no overlaps, no matter how big `n` is.

We'll learn it on **SAXPY** (Scaled `A` times `X` Plus `Y`): `y = a*x + y`, where `a` is a scalar. It's the "hello world" of numerical computing — the `S` is for single-precision, and the same operation in cuBLAS is literally called `cublasSaxpy`. Computationally it's just like vector add, so we can focus entirely on the loop pattern.

## Under the hood

Why launch fewer threads than elements? Because spawning a thread isn't free, and a thread that does *one* add then retires barely earns its keep:

- **Setup amortization.** Each thread pays a fixed cost to start: index math, register allocation, getting scheduled onto an SM. A grid-stride thread spreads that cost over dozens of elements instead of one.
- **Fewer block-scheduling overheads.** Millions of one-shot blocks all have to be handed out to the 40 SMs and torn down. A few thousand long-lived blocks keep the SMs continuously fed with far less churn — and avoid the "launch tail" where the last few straggler blocks run alone.
- **Latency hiding, tuned to the machine.** Each SM can hold many resident warps at once and instantly switch to a ready warp whenever one stalls on a slow memory load. You want *enough* blocks per SM to keep that pipeline full — a few blocks per SM is plenty — but not so many that you're just paying overhead. Sizing the grid to `multiProcessorCount` gives you exactly that knob.

So the recipe is: ask the device how many SMs it has, launch a small multiple of that many blocks, and let the loop handle the rest.

## A picture

```text
 grid = a few blocks per SM  →  total threads = gridDim.x * blockDim.x = STRIDE

 thread 0  ──▶ a[0] ───▶ a[0+S] ───▶ a[0+2S] ───▶ ...
 thread 1  ──▶ a[1] ───▶ a[1+S] ───▶ a[1+2S] ───▶ ...
 thread 2  ──▶ a[2] ───▶ a[2+S] ───▶ a[2+2S] ───▶ ...
   ...
 thread S-1 ▶ a[S-1] ─▶ a[2S-1] ─▶ ...

 array:  [0 1 2 ... S-1 | S S+1 ... 2S-1 | 2S ...]
          └─ pass 1 ──┘   └── pass 2 ──┘   └ pass 3
```

Each thread lands on `idx`, then `idx + stride`, then `idx + 2*stride`, … until it runs off the end. Notice adjacent threads still touch adjacent elements *within a pass* — that keeps memory accesses coalesced (more on that in exercise 04).

## Your task

Compute SAXPY: `y = a*x + y` for arrays of length `n = 1 << 25` (33M elements), updating `y` in place. The challenge: do it with a **fixed grid sized from the SM count** — *not* `ceil_div(n, block)` — so each thread visits multiple elements via a grid-stride loop.

Edit `saxpy.cu` and fill the `TODO`s:

1. The `__global__` kernel `saxpy` — compute your start index and the stride, then loop over the array doing `y[i] = a*x[i] + y[i]`.
2. The host function `solve` — query the device, pick a grid of a few blocks per SM (independent of `n`), and launch. Wrap your runtime calls in `CUDA_CHECK(...)`.

### The `solve` contract

```cpp
void solve(float a, const float* x, float* y, int n);
```

`x` and `y` are **device pointers** of length `n`; `a` is a plain host scalar (passed by value into the kernel — that's fine). `y` is read *and* written. The harness handles all allocation and copying.

> **No separate boundary `if` needed.** Unlike exercise 01, the loop condition `i < n` *is* the guard — a thread simply stops when it strides past the end. One clean loop covers everything.

## Functions & syntax you'll need

| Thing | Signature / form | What it does |
|-------|------------------|--------------|
| start index | `int idx = blockIdx.x * blockDim.x + threadIdx.x;` | This thread's first element. |
| stride | `int stride = gridDim.x * blockDim.x;` | Total threads in the grid = the jump between passes. |
| `gridDim.x` | built-in | Number of blocks in the grid (needed for the stride). |
| `blockDim.x` | built-in | Threads per block (needed for the stride). |
| `cudaGetDeviceProperties` | `cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int dev);` | Fills `p` with device info; use device `0`. |
| `cudaDeviceProp::multiProcessorCount` | `int` field | Number of SMs (40 on a T4) — base your grid on this. |
| `CUDA_CHECK(...)` | macro | Wraps a runtime call; on failure prints `file:line` + error string and aborts. |
| launch syntax | `saxpy<<<grid, block>>>(a, x, y, n);` | Launch with your fixed `grid` and `block`. |

Querying the machine looks like:

```cpp
cudaDeviceProp prop;
CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
int grid = prop.multiProcessorCount * SOME_SMALL_FACTOR;  // e.g. 32
```

> **Fun fact — `CUDA_CHECK`.** Every CUDA runtime call returns a `cudaError_t`, and silently ignoring it is the #1 cause of baffling CUDA bugs (an error from call A surfaces three calls later). The macro turns a silent failure into a loud `file:line` message. Kernel *launches* don't return an error directly, so the harness calls `CUDA_CHECK_KERNEL()` after your `solve` to catch those — you don't need to.

## How it's graded

Run `python grade.py`:

- **correctness** — `y == a*x + y` within tolerance (`max_abs_err <= 1e-3`).
- **efficiency** — SAXPY is memory-bound: read `x`, read `y`, write `y` ⇒ `3 * bytes` moved. You must hit **`bw_frac >= 0.55`** of theoretical peak. A correct grid-stride loop over a modest grid clears this with margin; launching too few blocks (starving the SMs) or a degenerate single block would tank it.
- **source** — your kernel must contain a real stride, so the grader checks the source for both `gridDim.x` and `blockDim.x`.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.

## Going deeper

- **One pattern, forever.** The grid-stride loop is the idiomatic shape for nearly every elementwise/streaming CUDA kernel. You'll reuse it in exercises 03, 05, and beyond. Internalize it now.
- **Try this:** change the per-SM factor from `32` to `1` (one block per SM) and re-grade. With only 40 blocks, each SM can't keep enough warps in flight to hide memory latency, and `gbps` drops — a hands-on demonstration of why occupancy matters.
- **Real cuBLAS** ships a hand-tuned `cublasSaxpy`; the grid-stride version you write here gets surprisingly close on a bandwidth-bound op like this one, because there's simply not much to optimize beyond "move bytes at full speed."
