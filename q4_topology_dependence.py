"""Experiment Q4: topology dependence across ER, WS, and BA networks."""
import argparse
import matplotlib.pyplot as plt

from physica_a_common import run_q4_topology, plot_q4_topology, DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "paper"], default="paper")
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--representative-d", type=float, default=0.26)
    parser.add_argument("--strong-alpha", type=float, default=0.90)
    parser.add_argument("--no-metric-grid", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()

    df = run_q4_topology(
        mode=args.mode,
        s=args.s,
        alpha=args.alpha,
        metric_grid=not args.no_metric_grid,
    )
    fig = plot_q4_topology(
        df,
        mode=args.mode,
        representative_d=args.representative_d,
        strong_alpha=args.strong_alpha,
    )
    if args.save_data:
        df.to_csv(DATA_DIR / f"q4_topology_{args.mode}.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
