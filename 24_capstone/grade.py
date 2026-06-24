import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="optimize.cu")
g.build()
m = g.run()
g.require_correct(m)
g.require_metric(m, "speedup", ">=", 3.0,   # T4-tuned, adjust if needed
                 "must beat the naive global-memory stencil by 3x")
g.require_metric(m, "bw_frac", ">=", 0.40,  # T4-tuned, adjust if needed
                 "achieved global-memory bandwidth utilization")
g.require_source(r"__shared__", present=True,
                 why="halo tiling in shared memory is the expected approach")
g.finish()
