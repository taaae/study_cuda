// Shared helpers used by every exercise harness.
// You generally don't need to read this to solve exercises, but it's short and
// worth understanding once: error checking, event timing, and device peak query.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---- Error checking -------------------------------------------------------
// Wrap every CUDA runtime call: CUDA_CHECK(cudaMalloc(...));
// On error it prints file:line and the human-readable message, then aborts.
#define CUDA_CHECK(call)                                                       \
  do {                                                                        \
    cudaError_t _err = (call);                                               \
    if (_err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n", __FILE__,        \
                   __LINE__, #call, cudaGetErrorString(_err));                \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// Call after a kernel launch to surface launch/async errors.
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                        \
    CUDA_CHECK(cudaGetLastError());                                           \
    CUDA_CHECK(cudaDeviceSynchronize());                                      \
  } while (0)

// ---- Small host helpers ---------------------------------------------------
__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---- Event-based timer ----------------------------------------------------
// GpuTimer t; t.start(); ...launch...; float ms = t.stop();
struct GpuTimer {
  cudaEvent_t a, b;
  GpuTimer() { cudaEventCreate(&a); cudaEventCreate(&b); }
  ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
  void start() { cudaEventRecord(a, 0); }
  float stop() {  // returns elapsed milliseconds
    cudaEventRecord(b, 0);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
  }
};

// Run fn() once to warm up, then `iters` times and return the best (min) ms.
// Using the minimum reduces noise from scheduling jitter on shared Colab GPUs.
template <typename F>
inline float time_kernel(F fn, int iters = 30) {
  fn();
  CUDA_CHECK_KERNEL();
  GpuTimer t;
  float best = 1e30f;
  for (int i = 0; i < iters; ++i) {
    t.start();
    fn();
    float ms = t.stop();
    if (ms < best) best = ms;
  }
  CUDA_CHECK_KERNEL();
  return best;
}

// ---- Device peak bandwidth (queried, not hardcoded) -----------------------
// Returns theoretical peak global-memory bandwidth in GB/s for the current
// device. DDR memory transfers on both clock edges -> factor of 2.
inline double peak_bandwidth_gbps() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  return 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
}

inline void print_device_banner() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("# device=%s sm_%d%d peak_bw=%.1f GB/s\n", p.name, p.major,
              p.minor, peak_bandwidth_gbps());
}

// ---- Metric reporting -----------------------------------------------------
// The harness prints results in a fixed, greppable format that grade.py parses:
//   RESULT correct=1
//   METRIC gbps=210.4
// Use these helpers so the format never drifts.
inline void report_correct(bool ok) { std::printf("RESULT correct=%d\n", ok ? 1 : 0); }
inline void report_metric(const char* name, double value) {
  std::printf("METRIC %s=%.6g\n", name, value);
}
