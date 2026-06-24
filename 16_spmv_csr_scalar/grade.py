import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="spmv.cu")
g.build()
m = g.run()
g.require_correct(m)
g.require_source(r"rowPtr", present=True,
                 why="you must index the CSR row pointers")
g.require_source(r"__shfl", present=False,
                 why="warp shuffle is exercise 17; here use plain per-row loops")
g.finish()
