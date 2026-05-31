"""Experiment Q3: open-minded depolarization phase diagram."""
import argparse
import matplotlib.pyplot as plt

from physica_a_common import run_q3_open_minded, plot_q3_open_minded, DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "paper"], default="paper")
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--positive-alpha", type=float, default=0.4)
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()

    df = run_q3_open_minded(
        mode=args.mode,
        s=args.s,
        positive_alpha=args.positive_alpha,
    )
    fig = plot_q3_open_minded(df, mode=args.mode)
    if args.save_data:
        df.to_csv(DATA_DIR / f"q3_open_minded_{args.mode}.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
