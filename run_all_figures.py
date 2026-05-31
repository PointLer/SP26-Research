"""Run all five experiment scripts.

Default mode is ``preview`` so that a new environment can be checked quickly.
Use ``--mode paper`` to recompute the full manuscript-resolution experiments.
"""
import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS_NOARGS = [
    "fig1a_hetero_equilibrium.py",
    "fig1b_homo_equilibrium.py",
]

SCRIPTS = [
    "q1_baseline_comparison.py",
    "q2_positive_phase_ba.py",
    "q3_open_minded_depolarization.py",
    "q4_topology_dependence.py",
    "q5_robustness_optional_appendix.py",
]

SCRIPTS_ALL = SCRIPTS_NOARGS + SCRIPTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "paper"], default="preview")
    parser.add_argument("--skip-q5", action="store_true", help="Skip optional appendix robustness checks.")
    parser.add_argument("--save-data", action="store_true", help="Also save CSV data tables.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    # Step 1: run standalone scripts (no command-line arguments)
    for script in SCRIPTS_NOARGS:
        print(f"Running {script} ...", flush=True)
        subprocess.run([sys.executable, str(root / script)], check=True)

    # Step 2: run experiment scripts with --mode and optional --save-data
    scripts = SCRIPTS[:-1] if args.skip_q5 else SCRIPTS
    if args.mode == "paper":
        print(
            "Paper mode is computationally heavy. Q2 alone runs 13 x 13 x 30 simulations.",
            flush=True,
        )
    for script in scripts:
        print(f"Running {script} in {args.mode} mode ...", flush=True)
        command = [sys.executable, str(root / script), "--mode", args.mode]
        if args.save_data:
            command.append("--save-data")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
