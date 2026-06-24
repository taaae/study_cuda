import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="streams.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed
g.require_metric(m, "speedup", ">=", 1.3, "overlap speedup over single-stream baseline")
g.require_source(r"cudaMemcpyAsync", present=True,
                 why="async copies are how transfers overlap compute")
g.require_source(r"cudaStreamCreate|cudaStream_t", present=True,
                 why="you need real streams to overlap")
g.finish()
