import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="gemm.cu", extra_flags=["-lcublas"])
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed
g.require_metric(m, "speedup", ">=", 1.5, "speedup over basic tiled GEMM")
g.require_metric(m, "frac_cublas", ">=", 0.30, "throughput vs cuBLAS")
g.require_source(r"float4", present=True,
                 why="use vectorized float4 global loads")
g.require_source(r"__shared__", present=True,
                 why="stage A and B tiles in shared memory")
g.finish()
