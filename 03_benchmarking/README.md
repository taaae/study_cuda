# Exercise 03 — Benchmarking with CUDA Events
> Measure GPU work honestly: warm up, time with CUDA events, take the best of N — and learn why a CPU stopwatch lies.

## The idea

Until now the harness has timed your kernels for you. But "how fast is my kernel?" is a question you'll ask constantly, and measuring it correctly is a genuine skill — get it wrong and every optimization decision you make afterward is built on noise.

Here's the trap. A kernel launch is **asynchronous**: `kernel<<<...>>>(...)` hands the work to the GPU and returns to the CPU *immediately*, long before the GPU finishes. So if you wrap a launch in a CPU timer like `clock()` or `std::chrono`, you measure the time it took to *enqueue* the work — microseconds — not the time the GPU spent doing it. Your "blazing fast" kernel was just fast to *submit*.

The fix is **CUDA events**: timestamps recorded *inside the GPU's stream of work*, so they capture real device execution. You drop one event before the kernel and one after, then ask CUDA for the elapsed time between them.

We practice on the simplest possible memory kernel — a **copy**, `out[i] = in[i]` — so all your attention goes to the timing, not the algorithm.

## Under the hood

A **stream** is an ordered queue of GPU work. When you call `cudaEventRecord(start)`, you're not recording the time *now* on the CPU — you're inserting a marker into that queue. The GPU stamps the marker with its clock when it *reaches* that point in the stream. Put one marker before the kernel and one after, and the gap between the two stamps is exactly the kernel's device time.

That's also why `cudaEventSynchronize(stop)` is essential: the CPU has to *wait* until the GPU has actually processed the `stop` marker before the elapsed time is meaningful. Without it you'd read a timestamp that hasn't happened yet.

Two more realities of measuring on real (and shared) hardware:

- **The first run is a liar.** The very first launch pays one-time costs: CUDA context creation, JIT compilation of the kernel, cold instruction/data caches, and GPU clocks that haven't boosted from idle yet. So you do a **warmup** — run it once, throw the number away.
- **Colab's T4 is shared and noisy.** Clock throttling and scheduler jitter mean repeated runs scatter. Run `iters` times and keep the **minimum**, not the average — the fastest run is the one *least* disturbed by interference, and it best reflects the kernel's true cost. (This is why `time_kernel` in `cuda_utils.cuh` returns the min, too.)

## A picture

```text
 CPU timeline:   launch returns instantly ──┐  (CPU clock sees ~0 here — WRONG)
                                            ▼
 GPU stream:  [ Record start ]→[  copy_kernel running...  ]→[ Record stop ]
                     │                                              │
                     └────────── cudaEventElapsedTime ─────────────┘
                                  = real device time (ms)

   warmup run   │ run 1 │ run 2 │ run 3 │ ... │ run N
   (discard) ───┘   take the MINIMUM of these  ───┘
```

## Your task

Two pieces, both in `bench.cu`:

1. The `__global__` kernel `copy_kernel` — a grid-stride copy, `out[i] = in[i]` (same shape as exercise 02).
2. The host function `benchmark_copy` — launch it, **time it yourself with CUDA events**, and return the **best elapsed milliseconds** over `iters` runs, after one warmup. On return, `out` must hold a correct copy of `in`.

The challenge: no using the library `time_kernel` — you write the event code by hand. That's the whole point.

### The `benchmark_copy` contract

```cpp
float benchmark_copy(const float* in, float* out, int n, int iters);
```

- `in`, `out` are **device pointers** of length `n`.
- Do **one warmup launch** (and a sync), then time `iters` launches and return the **minimum** elapsed time in **milliseconds**.
- After returning, `out == in` element-wise — make sure a real copy ran last.
- The student scaffold gives you `int block = 256;` and a `grid` to fill in; a `for` loop over `iters` with a `best` accumulator is already sketched.

## Functions & syntax you'll need

| Thing | Signature / form | What it does |
|-------|------------------|--------------|
| `cudaEvent_t` | `cudaEvent_t start, stop;` | Handle type for an event (a stream timestamp marker). |
| `cudaEventCreate` | `cudaError_t cudaEventCreate(cudaEvent_t* e);` | Allocate an event. |
| `cudaEventRecord` | `cudaError_t cudaEventRecord(cudaEvent_t e);` | Insert `e` into the stream — the GPU stamps it when it gets there. |
| `cudaEventSynchronize` | `cudaError_t cudaEventSynchronize(cudaEvent_t e);` | Block the CPU until the GPU has reached event `e`. |
| `cudaEventElapsedTime` | `cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b);` | Write the **milliseconds** between two recorded events into `*ms`. |
| `cudaEventDestroy` | `cudaError_t cudaEventDestroy(cudaEvent_t e);` | Free an event. |
| `cudaDeviceSynchronize` | `cudaError_t cudaDeviceSynchronize();` | Block until *all* prior GPU work is done — handy after the warmup. |
| launch syntax | `copy_kernel<<<grid, block>>>(in, out, n);` | Launch the copy. |
| `ceil_div` | `int ceil_div(int a, int b)` | From `cuda_utils.cuh`; fine for sizing the grid, though grid-stride means it needn't equal `ceil_div(n, block)`. |

The core timing dance for one iteration:

```cpp
cudaEventRecord(start);
copy_kernel<<<grid, block>>>(in, out, n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);              // wait for the GPU to reach 'stop'
float ms = 0.f;
cudaEventElapsedTime(&ms, start, stop);  // milliseconds between the events
```

## How it's graded

A copy moves `2 * bytes` (read `in`, write `out`). Achieved bandwidth is `(2 * n * 4) / (ms * 1e-3) / 1e9` GB/s, divided by `peak_bandwidth_gbps()` to get the fraction of peak. Run `python grade.py`:

- **correctness** — `out == in` exactly after `benchmark_copy` returns (`max_abs_err == 0`).
- **ms_ratio** — the harness *also* times your copy with its own `time_kernel`, and your returned `ms` must agree within ~25%: `ms_ratio` (= your ms / harness ms) must land in **`[0.6, 1.6]`**. Forget the warmup or report the *average* instead of the *min* and your number drifts out of this band — this check is what forces honest methodology.
- **efficiency** — `bw_frac` from *your* `ms` must be **`>= 0.70`**. A pure copy is the easiest memory-bound kernel there is, so the threshold is deliberately strict; missing it means your timing is wrong or your grid starves the SMs.
- **source** — you must use `cudaEventRecord` *and* `cudaEventElapsedTime` (proof you wrote the event code, not a CPU timer).

Run `python grade.py --check-solution` to grade the reference solution instead of yours.

## Going deeper

- **A copy is the bandwidth benchmark.** Because it does zero arithmetic, `copy_kernel` measures pure memory throughput — it's the standard way to find a card's *realistic* peak (always a bit below the theoretical number the datasheet prints). On a T4 expect well over 200 GB/s.
- **Events do more than timing.** The same `cudaEvent` objects let you synchronize *between* streams (wait for work in stream A before starting stream B), the foundation of overlapping copies with compute.
- **Try this:** delete your warmup launch and re-grade. Watch `ms_ratio` slip out of band as the first cold run contaminates your minimum.
