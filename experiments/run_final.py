"""Run the displayed 13/29/31 protocol from scratch in a separate output tree.

The displayed numbers summarize seeds 13, 29, and 31. Read
results/final/README.md before interpreting the means.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_experiments.py"),
                "--config",
                str(ROOT / "experiments" / "final_study.yaml"),
                *sys.argv[1:],
            ],
            cwd=ROOT,
        )
    )
