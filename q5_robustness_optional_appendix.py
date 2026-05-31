"""Experiment Q5: optional robustness and scalability checks.

This script is kept separate because the current manuscript version
treats Q5 as optional/appendix material.
"""
import argparse
import matplotlib.pyplot as plt

from physica_a_common import run_q5_robustness, plot_q5_robustness, DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "paper"], default="paper")
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()

    df = run_q5_robustness(mode=args.mode, s=args.s, alpha=args.alpha)
    fig = plot_q5_robustness(df, mode=args.mode)
    if args.save_data:
        df.to_csv(DATA_DIR / f"q5_robustness_optional_appendix_{args.mode}.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
