import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="cg.cu")
g.build()
m = g.run()
g.require_correct(m)
# Relative residual recomputed by the harness must hit the solve tolerance.
g.require_metric(m, "rel_resid", "<=", 1e-3, "CG converged (relative residual)")  # T4-tuned, adjust if needed
# Capstone source checks are lenient: must have a CSR SpMV loop and a reduction.
g.require_source(r"rowPtr", present=True, why="CG needs an SpMV over the CSR matrix")
g.require_source(r"atomicAdd|__shfl_down_sync|__syncthreads",
                 present=True, why="dot products need an on-device reduction")
g.finish()
