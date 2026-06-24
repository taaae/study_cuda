import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="scan.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed. Scan is multi-pass, so the bar is lenient.
g.require_metric(m, "bw_frac", ">=", 0.30, "global-memory bandwidth utilization")
g.require_source(r"__shared__", present=True, why="per-block scan must live in shared memory")
g.require_source(r"__syncthreads", present=True, why="up/down-sweep needs block barriers")
g.finish()
