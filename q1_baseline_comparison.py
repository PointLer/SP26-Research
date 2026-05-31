"""Experiment Q1: baseline mechanism comparison.

Compares classical DW, symmetric biased DW, and asymmetric biased DW
under the same graph, initial opinions, and edge schedule.
"""
import argparse
import matplotlib.pyplot as plt

from physica_a_common import run_q1_baseline, plot_q1_baseline, DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "paper"], default="paper")
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()

    df, exemplars = run_q1_baseline(mode=args.mode, s=args.s, alpha=args.alpha)
    fig, summary = plot_q1_baseline(df, exemplars, mode=args.mode)
    if args.save_data:
        df.to_csv(DATA_DIR / f"q1_baseline_{args.mode}.csv", index=False)
        summary.to_csv(DATA_DIR / f"q1_baseline_summary_{args.mode}.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
