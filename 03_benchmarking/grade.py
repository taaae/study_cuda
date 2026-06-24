import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="bench.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed
g.require_metric(m, "bw_frac", ">=", 0.70, "achieved copy bandwidth (fraction of peak)")
g.require_metric(m, "ms_ratio", ">=", 0.6, "your timing must agree with the harness timer")
g.require_metric(m, "ms_ratio", "<=", 1.6, "your timing must agree with the harness timer")
g.require_source(r"cudaEventRecord", present=True,
                 why="time the kernel with cudaEventRecord, not a CPU clock")
g.require_source(r"cudaEventElapsedTime", present=True,
                 why="read elapsed time with cudaEventElapsedTime")
g.finish()
