import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="occupancy.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed
g.require_metric(m, "bw_frac", ">=", 0.55, "global-memory bandwidth utilization")
g.require_source(r"cudaOccupancyMaxPotentialBlockSize", present=True,
                 why="pick the launch config with the occupancy API, not by hand")
g.finish()
