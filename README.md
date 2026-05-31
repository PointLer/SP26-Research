# Asymmetric Biased Assimilation in the Deffuant Weisbuch Model Figure Scripts
This repository contains the  figure-generation codes split by experiment.

## Files

- `fig1a_hetero_equilibrium.py`: Figure 1(a), pairwise contraction region for intermediate equilibria in $(b,c)$ space.
- `fig1b_homo_equilibrium.py`: Figure 1(b), pairwise contraction region for the symmetric neutral equilibrium in $(b,\mu)$ space.
- `physica_a_common.py`: shared DW update rule, simulation utilities, metrics, and plotting helpers.
- `q1_baseline_comparison.py`: Figure Q1, mechanism comparison.
- `q2_positive_phase_ba.py`: Figure Q2, positive-parameter phase diagram on a BA network.
- `q3_open_minded_depolarization.py`: Figure Q3, open-minded depolarization phase diagram.
- `q4_topology_dependence.py`: Figure Q4, topology dependence across ER/WS/BA.
- `q5_robustness_optional_appendix.py`: optional Figure Q5 robustness checks.
- `run_all_figures.py`: runs all figures; default is quick `preview` mode.

## Run

Run one experiment:

```bash
python q1_baseline_comparison.py --mode paper
```

Quickly test all experiments:

```bash
python run_all_figures.py
```

Recompute the full paper-mode figures:

```bash
python run_all_figures.py --mode paper
```

Paper mode is slow because Q2 alone runs `13 x 13 x 30 = 5070`
simulations. A `KeyboardInterrupt` or exit code `0xC000013A` means the
process was stopped manually, not that the formula is wrong.

If only the main manuscript figures are needed and Q5 is treated as appendix
material, use:

```bash
python run_all_figures.py --mode paper --skip-q5
```

Figures are saved in this folder as `.png`, `.svg`, and `.pdf`.
By default, the scripts only save figures. To also save CSV data tables in
`data/`, add `--save-data`.

## Default paper parameters

- Number of agents: `N = 1000`
- Monte Carlo runs: `K = 30`
- Maximum edge updates: `T_max = 300000`
- Convergence parameter: `mu = 0.3`
- Total positive bias strength: `s = (b + c) / 2 = 0.5`
- Average network degree: approximately 6
- Cluster tolerance: `epsilon_c = 1e-3`
- Q1/Q3/Q4/Q5 default positive asymmetry: `alpha = 0.4`
- Q4 plotted numerical slices: representative confidence threshold `d = 0.26`, strong positive asymmetry `alpha = 0.90`
