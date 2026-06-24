import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grader import Grader

# Grid sync / cooperative launch requires relocatable device code.
g = Grader(student="normalize.cu", extra_flags=["-rdc=true"])
g.build()
m = g.run()
g.require_correct(m)
g.require_source(r"grid\.sync", present=True,
                 why="must do a grid-wide barrier with cooperative groups")
g.require_source(r"cudaLaunchCooperativeKernel", present=True,
                 why="must launch with cudaLaunchCooperativeKernel, not <<< >>>")
g.finish()
