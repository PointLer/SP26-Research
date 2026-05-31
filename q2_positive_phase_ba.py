"""Experiment Q2: positive-parameter phase diagram on a BA network."""
import argparse
import matplotlib.pyplot as plt

from physica_a_common import run_q2_phase_diagram, plot_q2_phase, DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "paper"], default="paper")
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--network", choices=["ba", "er", "ws"], default="ba")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()

    df = run_q2_phase_diagram(mode=args.mode, s=args.s, network=args.network)
    fig = plot_q2_phase(df, mode=args.mode, network=args.network)
    if args.save_data:
        df.to_csv(DATA_DIR / f"q2_phase_{args.network}_{args.mode}.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
