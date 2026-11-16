"""Train both conditions x all seeds in parallel (one process each).

    python run_all.py [steps] [seeds] [standard|hard]
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    steps = sys.argv[1] if len(sys.argv) > 1 else "400000"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    scenario = sys.argv[3] if len(sys.argv) > 3 else "standard"
    procs = []
    for condition in ("base", "deadline_aware"):
        for seed in range(n_seeds):
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(SCRIPT_DIR, "train.py"),
                 "--condition", condition, "--seed", str(seed), "--steps", steps, "--scenario", scenario],
                cwd=SCRIPT_DIR))
    for p in procs:
        p.wait()
    print("all runs finished")


if __name__ == "__main__":
    main()
