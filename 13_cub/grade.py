import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="scan_cub.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed. CUB's scan is tuned per-arch and far outruns the
# hand-rolled scan of exercise 10 (which only needed bw_frac >= 0.30).
g.require_metric(m, "bw_frac", ">=", 0.55, "global-memory bandwidth utilization")
g.require_source(r"cub::DeviceScan", present=True,
                 why="must use CUB's device-wide scan primitive")
g.finish()
