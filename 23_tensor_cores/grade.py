import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="wmma_gemm.cu")
g.build()
m = g.run()
g.require_correct(m)
g.require_metric(m, "gflops", ">=", 12000,  # T4-tuned, adjust if needed
                 "must run on tensor cores (>12 TFLOPS, above FP32 peak)")
g.require_source(r"wmma::fragment", present=True, why="must use WMMA fragments")
g.require_source(r"mma_sync", present=True, why="must do the MMA with mma_sync")
g.finish()
