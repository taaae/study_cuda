import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="histogram.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed. Privatization should comfortably beat the
# naive global-atomic baseline on a 256-bin histogram.
g.require_metric(m, "speedup", ">=", 2.0, "speedup over naive global-atomic baseline")
g.require_source(r"__shared__", present=True, why="private histogram must live in shared memory")
g.require_source(r"atomicAdd", present=True, why="counting collisions requires atomics")
g.finish()
