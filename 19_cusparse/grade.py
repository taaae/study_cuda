import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

# cuSPARSE must be linked.
g = Grader(student="cusparse_spmv.cu", extra_flags=["-lcusparse"])
g.build()
m = g.run()
g.require_correct(m)
# No hard speedup threshold — correctness via the library is the bar. We do
# require the SpMV actually ran and produced a finite timing.  # T4-tuned, adjust if needed
g.require_metric(m, "ms", ">", 0.0, "cuSPARSE SpMV actually ran (timed)")
g.require_source(r"cusparseSpMV", present=True,
                 why="must call the modern generic cusparseSpMV")
g.require_source(r"cusparseScsrmv", present=False,
                 why="legacy cusparseScsrmv is deprecated — use the generic API")
g.finish()
