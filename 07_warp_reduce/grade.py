import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="reduce.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed
g.require_metric(m, "bw_frac", ">=", 0.50, "global-memory bandwidth utilization")
g.require_source(r"__shfl_down_sync", present=True,
                 why="intra-warp reduction must use warp shuffle, not shared memory")
g.finish()
