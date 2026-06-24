"""Shared grading utilities used by every exercise's grade.py.

A grade.py typically does:

    from grader import Grader
    g = Grader(student="vadd.cu", harness="harness.cu", arch="sm_75")
    g.build()                       # nvcc compile, hard-fail on error
    m = g.run()                     # run binary, parse RESULT/METRIC lines
    g.require_correct(m)
    g.require_metric(m, "bw_frac", ">=", 0.70, "global bandwidth utilization")
    g.require_source(r"__shfl_down_sync", present=True,
                     why="must use warp shuffle, not shared memory")
    g.finish()

Run any grade.py with `--check-solution` to grade the reference solution in
solutions/ instead of your file — use it to confirm the thresholds are sane.
"""
import argparse
import os
import re
import subprocess
import sys

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class GradeError(Exception):
    pass


class Grader:
    def __init__(self, student, harness="harness.cu", arch="sm_75",
                 extra_flags=None, run_args=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--check-solution", action="store_true",
                        help="grade solutions/<student> instead of your file")
        ap.add_argument("--keep", action="store_true", help="keep the built binary")
        args, _ = ap.parse_known_args()

        self.here = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.common = os.path.join(self.here, "..", "common")
        self.harness = harness
        self.arch = arch
        self.extra_flags = extra_flags or []
        self.run_args = run_args or []
        self.keep = args.keep
        self.use_solution = args.check_solution

        self.student_name = student
        if self.use_solution:
            self.student_path = os.path.join(self.here, "solutions", student)
        else:
            self.student_path = os.path.join(self.here, student)
        self.binary = os.path.join(self.here, "_grade_bin")
        self.checks = []  # (ok, label, detail)

        mode = "REFERENCE SOLUTION" if self.use_solution else "your solution"
        print(f"{DIM}# grading {mode}: {self.student_name}{RESET}")

    # -- build ---------------------------------------------------------------
    def build(self):
        if not os.path.exists(self.student_path):
            self._fail_hard(f"file not found: {self.student_path}")
        # No shell here (subprocess uses a list), so pass the quotes literally:
        # nvcc must receive  -DSOLUTION_FILE="/abs/path.cu"  so the macro expands
        # to a valid string literal usable in  #include SOLUTION_FILE.
        sol_macro = f'-DSOLUTION_FILE="{self.student_path}"'
        cmd = ["nvcc", "-O3", f"-arch={self.arch}", "-std=c++17",
               f"-I{self.common}", sol_macro,
               os.path.join(self.here, self.harness), "-o", self.binary]
        cmd += self.extra_flags
        print(f"{DIM}# {' '.join(cmd)}{RESET}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            self._fail_hard("compilation failed:\n" + r.stderr)
        if r.stderr.strip():
            print(f"{YELLOW}{r.stderr.strip()}{RESET}")

    # -- run -----------------------------------------------------------------
    def run(self):
        r = subprocess.run([self.binary, *map(str, self.run_args)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            self._fail_hard(f"program crashed (exit {r.returncode}):\n"
                            + r.stdout + r.stderr)
        metrics = {}
        correct = None
        for line in r.stdout.splitlines():
            if line.startswith("#"):
                print(f"{DIM}{line}{RESET}")
            m = re.match(r"METRIC (\S+)=(\S+)", line)
            if m:
                metrics[m.group(1)] = float(m.group(2))
            m = re.match(r"RESULT correct=(\d)", line)
            if m:
                correct = m.group(1) == "1"
        metrics["_correct"] = correct
        if not self.keep:
            try:
                os.remove(self.binary)
            except OSError:
                pass
        return metrics

    # -- checks --------------------------------------------------------------
    def require_correct(self, metrics):
        ok = metrics.get("_correct") is True
        self.checks.append((ok, "correctness",
                            "output matches reference" if ok
                            else "output does NOT match reference"))

    _OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b, "<": lambda a, b: a < b}

    def require_metric(self, metrics, key, op, thr, label):
        if key not in metrics:
            self.checks.append((False, label, f"metric '{key}' missing"))
            return
        val = metrics[key]
        ok = self._OPS[op](val, thr)
        self.checks.append((ok, label, f"{key}={val:.4g} (need {op} {thr})"))

    def require_source(self, pattern, present=True, why=""):
        with open(self.student_path) as f:
            src = strip_comments(f.read())
        found = re.search(pattern, src) is not None
        ok = found == present
        verb = "must use" if present else "must NOT use"
        self.checks.append((ok, f"source: {verb} /{pattern}/", why))

    # -- finish --------------------------------------------------------------
    def finish(self):
        print()
        allok = True
        for ok, label, detail in self.checks:
            tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
            print(f"  [{tag}] {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
            allok = allok and ok
        print()
        if allok:
            print(f"{GREEN}*** ALL CHECKS PASSED ***{RESET}")
            sys.exit(0)
        else:
            print(f"{RED}*** SOME CHECKS FAILED ***{RESET}  "
                  f"{DIM}(see hints.md if you're stuck){RESET}")
            sys.exit(1)

    def _fail_hard(self, msg):
        print(f"{RED}FATAL: {msg}{RESET}")
        sys.exit(2)


def strip_comments(src):
    """Remove // and /* */ comments so source checks don't match commented code."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return src
