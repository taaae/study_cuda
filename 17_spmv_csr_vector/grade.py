import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="spmv.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed
g.require_metric(m, "speedup", ">=", 1.5,
                 "warp-per-row vs scalar baseline on the imbalanced matrix")
g.require_source(r"__shfl_down_sync", present=True,
                 why="reduce the lane partials with a warp shuffle, not shared memory")
g.finish()
