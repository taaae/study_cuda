import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="spmv_ell.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed. On the near-uniform matrix ELL's coalesced,
# load-balanced access should comfortably beat scalar CSR.
g.require_metric(m, "speedup", ">=", 1.2, "ELL vs scalar-CSR speedup")
g.require_source(r"k\s*\*\s*nrows", present=True,
                 why="must use the column-major stride k*nrows to stay coalesced")
g.require_source(r"rowPtr", present=False,
                 why="ELL has no rowPtr — every row has exactly maxnnz slots")
g.finish()
