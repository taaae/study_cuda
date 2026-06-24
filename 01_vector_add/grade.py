import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="vadd.cu")
g.build()
m = g.run()
g.require_correct(m)
g.require_metric(m, "bw_frac", ">=", 0.55, "global-memory bandwidth utilization")
g.require_source(r"<<<", present=True, why="launch the kernel with <<<grid, block>>>")
g.finish()
