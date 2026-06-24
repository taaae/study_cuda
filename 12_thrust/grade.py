import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

# --extended-lambda lets thrust placeholders / device lambdas compile.
g = Grader(student="compact.cu", extra_flags=["--extended-lambda"])
g.build()
m = g.run()
g.require_correct(m)
# No strict perf threshold: Thrust is the point. ms is reported for reference.
g.require_source(r"thrust::", present=True, why="this exercise is about using Thrust")
g.require_source(r"copy_if|transform_reduce", present=True,
                 why="must use a Thrust algorithm (copy_if and/or transform_reduce)")
g.finish()
