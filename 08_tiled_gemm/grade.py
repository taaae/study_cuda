import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="gemm.cu", extra_flags=["-lcublas"])
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed
g.require_metric(m, "frac_cublas", ">=", 0.12, "throughput vs cuBLAS")
g.require_source(r"__shared__", present=True,
                 why="tile A and B through shared memory to raise arithmetic intensity")
g.finish()
