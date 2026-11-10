\
\


from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PY = sys.executable
SEEDS = str(ROOT / "opt_seeds.npz")


def build_cases(parallel: int | None):
    common = []
    if parallel is not None:
        common += ["--parallel", str(parallel)]

    early = {
        "baseline_es": [
            PY, "optimizer_baseline.py", "--seeds", SEEDS, "--use-early-stop", *common
        ],
        "a1_es": [
            PY, "optimizer.py", "--seeds", SEEDS,
            "--no-de", "--no-pso", "--no-opt-perm-align",
            "--use-early-stop", *common
        ],
        "a2_es": [
            PY, "optimizer.py", "--seeds", SEEDS,
            "--use-de", "--no-pso", "--no-opt-perm-align",
            "--use-early-stop", *common
        ],
        "a3_es": [
            PY, "optimizer.py", "--seeds", SEEDS,
            "--use-de", "--use-pso", "--no-opt-perm-align",
            "--use-early-stop", *common
        ],
        "a4_es": [
            PY, "optimizer.py", "--seeds", SEEDS,
            "--use-de", "--use-pso", "--use-opt-perm-align",
            "--use-early-stop", *common
        ],
    }

    fixed = {
        "baseline_fixed": [
            PY, "optimizer_baseline.py", "--seeds", SEEDS, "--no-early-stop", *common
        ],
        "a1_fixed": [
            PY, "optimizer.py", "--seeds", SEEDS,
            "--no-de", "--no-pso", "--no-opt-perm-align",
            "--no-early-stop", *common
        ],
        "a2_fixed": [
            PY, "optimizer.py", "--seeds", SEEDS,
            "--use-de", "--no-pso", "--no-opt-perm-align",
            "--no-early-stop", *common
        ],
        "a3_fixed": [
            PY, "optimizer.py", "--seeds", SEEDS,
            "--use-de", "--use-pso", "--no-opt-perm-align",
            "--no-early-stop", *common
        ],
        "a4_fixed": [
            PY, "optimizer.py", "--seeds", SEEDS,
            "--use-de", "--use-pso", "--use-opt-perm-align",
            "--no-early-stop", *common
        ],
    }
    return early | fixed


def main():
    parser = argparse.ArgumentParser(description="Run aligned ablation experiments.")
    parser.add_argument("--parallel", type=int, default=None, help="pass through to optimizer scripts")
    parser.add_argument("--execute", action="store_true", help="actually run commands")
    parser.add_argument("--group", choices=["all", "early", "fixed"], default="all")
    parser.add_argument("--case", action="append", default=None, help="run only selected case names")
    args = parser.parse_args()

    cases = build_cases(args.parallel)
    if args.group == "early":
        cases = {k: v for k, v in cases.items() if k.endswith("_es")}
    elif args.group == "fixed":
        cases = {k: v for k, v in cases.items() if k.endswith("_fixed")}

    if args.case:
        wanted = set(args.case)
        cases = {k: v for k, v in cases.items() if k in wanted}

    for name, cmd in cases.items():
        print(f"[{name}]")
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        if args.execute:
            subprocess.run(cmd, cwd=ROOT, check=True)
        print()


if __name__ == "__main__":
    main()
