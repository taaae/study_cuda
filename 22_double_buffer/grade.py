import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="gemm.cu")
g.build()
m = g.run()
g.require_correct(m)
g.require_metric(m, "speedup", ">=", 1.10,  # T4-tuned, adjust if needed
                 "double-buffering must beat single-buffer tiled GEMM")
g.require_source(r"__shared__", present=True, why="must tile through shared memory")
# Two shared tile buffers in `__shared__ float As[2][...][...]` style.
g.require_source(r"\[\s*2\s*\]\s*\[", present=True,
                 why="must declare TWO shared buffers (ping-pong), e.g. As[2][TILE][TILE]")
g.finish()
