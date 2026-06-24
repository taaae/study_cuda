import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="transpose.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed
g.require_metric(m, "speedup", ">=", 1.3, "must beat the doubly-strided naive baseline")
g.require_metric(m, "bw_frac", ">=", 0.35, "achieved bandwidth (one strided side caps this)")
g.require_source(r"threadIdx\.x", present=True,
                 why="map threads to elements with threadIdx.x for coalescing")
g.require_source(r"blockIdx", present=True,
                 why="use a 2-D grid with blockIdx for the transpose")
g.require_source(r"__shared__", present=False,
                 why="no shared memory in this exercise (that is exercise 05)")
g.finish()
