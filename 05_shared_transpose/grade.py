import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

g = Grader(student="transpose.cu")
g.build()
m = g.run()
g.require_correct(m)
# T4-tuned, adjust if needed
g.require_metric(m, "speedup", ">=", 1.5, "shared-memory tiling must beat the single-sided naive")
g.require_metric(m, "bw_frac", ">=", 0.60, "both sides coalesced + no bank conflicts")
g.require_source(r"__shared__", present=True,
                 why="stage the tile in shared memory")
g.require_source(r"__syncthreads", present=True,
                 why="barrier after filling the tile, before reading it back")
g.require_source(r"TILE\s*\+\s*1", present=True,
                 why="pad the shared tile (TILE + 1) to avoid 32-way bank conflicts")
g.finish()
