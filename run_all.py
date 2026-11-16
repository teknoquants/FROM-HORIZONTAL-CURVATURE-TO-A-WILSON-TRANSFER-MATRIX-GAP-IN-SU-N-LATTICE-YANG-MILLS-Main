# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Run every validation script of the suite and print a summary. Exit code 0 only if every check passes."""
import subprocess
import sys
import time

SCRIPTS = ["validate_part1_lsi.py", "validate_part2_oneloop.py", "validate_part3_decay.py", "validate_part4_gap.py"]


def main():
    results = []
    for s in SCRIPTS:
        print(f"=== {s}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, s])
        results.append((s, r.returncode == 0, time.time() - t0))
        print(flush=True)
    print("=== Summary")
    for s, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {s}  ({dt:.0f} s)")
    allok = all(ok for _, ok, _ in results)
    print("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED")
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
