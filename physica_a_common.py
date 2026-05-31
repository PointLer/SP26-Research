# Generated from PhysicaA_five_figures_standalone.ipynb.
# Font size and plotting style match the latest manuscript figures.

"""
Physica A five-question simulation toolkit for the asymmetric biased DW model.

This module implements the five experiment blocks requested in
PhysicaA_five_research_questions.md:
Q1 baseline mechanism comparison, Q2 positive-parameter phase diagrams,
Q3 open-minded depolarization, Q4 topology dependence, and Q5 robustness.

The update rule is the fractional biased-assimilation DW formula used in the
paper. All comparative experiments can reuse the same graph, initial opinions,
and edge-sampling schedule, so differences are attributable to the model or
parameter being tested.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from numba import njit


ROOT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = ROOT_DIR
FIG_DIR = ROOT_DIR
DATA_DIR = ROOT_DIR / "data"
for _p in (OUTPUT_DIR, FIG_DIR, DATA_DIR):
    _p.mkdir(parents=True, exist_ok=True)




def set_plot_style() -> None:
    """Submission figure style with 20 pt text before LaTeX scaling.

    The target is that labels, ticks, and legends appear close to the caption
    size after the figures are inserted into the manuscript.
    """
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 450,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.titleweight": "bold",
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.fontsize": 20,
        "legend.title_fontsize": 20,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 2.8,
        "lines.markersize": 9,
    })


set_plot_style()


@dataclass(frozen=True)
class SimConfig:
    n: int = 1000
    mu: float = 0.3
    d: float = 0.25
    s: float = 0.5
    alpha: float = 0.4
    p_open: float = 0.0
    model: str = "asymmetric"
    network: str = "ba"
    avg_degree: int = 6
    ba_m: int = 3
    ws_rewire: float = 0.1
    max_steps: int = 300_000
    check_interval: int = 1000
    patience_checks: int = 8
    tol: float = 1e-7
    eps: float = 1e-6
    cluster_tol: float = 1e-3
    seed: int = 42
    open_low: float = -0.8
    open_high: float = -0.2
    positive_alpha_for_open: float = 0.4


PRESETS: Dict[str, Dict[str, object]] = {
    "preview": {
        "n": 300,
        "k": 3,
        "max_steps": 60_000,
        "d_values": np.linspace(0.08, 0.48, 6),
        "alpha_values": np.linspace(-0.8, 0.8, 7),
        "p_open_values": np.linspace(0.0, 1.0, 6),
        "n_values": [300, 600, 1000],
    },
    "paper": {
        "n": 1000,
        "k": 30,
        "max_steps": 300_000,
        "d_values": np.round(np.linspace(0.02, 0.50, 13), 3),
        "alpha_values": np.round(np.linspace(-0.9, 0.9, 13), 3),
        "p_open_values": np.round(np.linspace(0.0, 1.0, 11), 3),
        "n_values": [500, 1000, 3000, 5000],
    },
}


@njit
def _clip01(x: float, eps: float) -> float:
    if x < eps:
        return eps
    if x > 1.0 - eps:
        return 1.0 - eps
    return x


@njit
def _dw_fractional_core(
    x0: np.ndarray,
    edges: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    edge_schedule: np.ndarray,
    mu: float,
    d: float,
    eps: float,
    tol: float,
    check_interval: int,
    patience_checks: int,
) -> Tuple[np.ndarray, int, bool]:
    """Numba-accelerated fractional DW dynamics with interval-based stopping."""
    x = x0.copy()
    last = x0.copy()
    stable_checks = 0
    n_steps = len(edge_schedule)

    for t in range(n_steps):
        e_idx = edge_schedule[t]
        u = edges[e_idx, 0]
        v = edges[e_idx, 1]

        xi = _clip01(x[u], eps)
        xj = _clip01(x[v], eps)

        if abs(xi - xj) <= d:
            bu = b[u]
            cu = c[u]
            bv = b[v]
            cv = c[v]

            xi_b = xi ** bu
            xi_c = (1.0 - xi) ** cu
            xj_b = xj ** bv
            xj_c = (1.0 - xj) ** cv

            den_u = (1.0 - mu) + mu * xj * xi_b + mu * (1.0 - xj) * xi_c
            den_v = (1.0 - mu) + mu * xi * xj_b + mu * (1.0 - xi) * xj_c

            if den_u > eps:
                new_xi = ((1.0 - mu) * xi + mu * xj * xi_b) / den_u
            else:
                new_xi = xi
            if den_v > eps:
                new_xj = ((1.0 - mu) * xj + mu * xi * xj_b) / den_v
            else:
                new_xj = xj

            x[u] = _clip01(new_xi, eps)
            x[v] = _clip01(new_xj, eps)

        if (t + 1) % check_interval == 0:
            max_delta = 0.0
            for i in range(x.shape[0]):
                delta = abs(x[i] - last[i])
                if delta > max_delta:
                    max_delta = delta
                last[i] = x[i]

            if max_delta < tol:
                stable_checks += 1
            else:
                stable_checks = 0

            if stable_checks >= patience_checks:
                return x, t + 1, True

    return x, n_steps, False


def make_graph(config: SimConfig) -> nx.Graph:
    """Create ER/WS/BA networks with controlled average degree."""
    network = config.network.lower()
    if network == "ba":
        m = max(1, int(round(config.avg_degree / 2))) if config.ba_m is None else int(config.ba_m)
        return nx.barabasi_albert_graph(config.n, m, seed=config.seed)
    if network == "er":
        p = min(1.0, float(config.avg_degree) / max(config.n - 1, 1))
        return nx.erdos_renyi_graph(config.n, p, seed=config.seed)
    if network == "ws":
        k = int(config.avg_degree)
        if k % 2 == 1:
            k += 1
        return nx.watts_strogatz_graph(config.n, k, config.ws_rewire, seed=config.seed)
    raise ValueError(f"Unknown network: {config.network}")


def graph_to_edges(G: nx.Graph) -> np.ndarray:
    edges = np.asarray(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        raise ValueError("The graph has no edges.")
    return edges


def initial_opinions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=n).astype(np.float64)


def edge_schedule(num_edges: int, max_steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_edges, size=max_steps, dtype=np.int64)


def build_bias_arrays(config: SimConfig, seed_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed + seed_offset)
    n = config.n
    model = config.model.lower()

    if model == "classical":
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    if model == "symmetric":
        return np.full(n, config.s, dtype=np.float64), np.full(n, config.s, dtype=np.float64)

    if model == "asymmetric":
        b_val = config.s * (1.0 + config.alpha)
        c_val = config.s * (1.0 - config.alpha)
        return np.full(n, b_val, dtype=np.float64), np.full(n, c_val, dtype=np.float64)

    if model == "open_minded":
        b_val = config.s * (1.0 + config.positive_alpha_for_open)
        c_val = config.s * (1.0 - config.positive_alpha_for_open)
        b = np.full(n, b_val, dtype=np.float64)
        c = np.full(n, c_val, dtype=np.float64)
        mask = rng.random(n) < config.p_open
        if mask.any():
            b[mask] = rng.uniform(config.open_low, config.open_high, size=mask.sum())
            c[mask] = rng.uniform(config.open_low, config.open_high, size=mask.sum())
        return b, c

    raise ValueError(f"Unknown model: {config.model}")


def _cluster_sizes(sorted_x: np.ndarray, cluster_tol: float) -> List[int]:
    sizes: List[int] = []
    start = 0
    for i in range(1, len(sorted_x)):
        if sorted_x[i] - sorted_x[i - 1] > cluster_tol:
            sizes.append(i - start)
            start = i
    sizes.append(len(sorted_x) - start)
    return sizes


def compute_metrics(x: np.ndarray, cluster_tol: float = 1e-3) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    xs = np.sort(x)
    sizes = _cluster_sizes(xs, cluster_tol)
    c_count = len(sizes)
    s_max = max(sizes) / len(x)
    return {
        "P": float(4.0 * np.var(x)),
        "C": float(c_count),
        "D": float(2.0 * np.mean(x) - 1.0),
        "R_extreme": float(np.mean((x < 0.05) | (x > 0.95))),
        "R_neutral": float(np.mean((x >= 0.4) & (x <= 0.6))),
        "S_max_over_N": float(s_max),
        "Pr_consensus_proxy": float(c_count == 1 or s_max >= 0.95),
    }


def simulate_once(
    config: SimConfig,
    G: Optional[nx.Graph] = None,
    x0: Optional[np.ndarray] = None,
    schedule: Optional[np.ndarray] = None,
    seed_offset: int = 0,
) -> Dict[str, object]:
    """Run one simulation and return final opinions plus order parameters."""
    cfg = replace(config, seed=config.seed + seed_offset)
    if G is None:
        G = make_graph(cfg)
    edges = graph_to_edges(G)
    if x0 is None:
        x0 = initial_opinions(cfg.n, cfg.seed)
    if schedule is None:
        schedule = edge_schedule(len(edges), cfg.max_steps, cfg.seed + 10_000)

    b, c = build_bias_arrays(cfg, seed_offset=20_000)
    final_x, steps, converged = _dw_fractional_core(
        np.asarray(x0, dtype=np.float64),
        edges,
        b,
        c,
        schedule,
        cfg.mu,
        cfg.d,
        cfg.eps,
        cfg.tol,
        cfg.check_interval,
        cfg.patience_checks,
    )
    metrics = compute_metrics(final_x, cluster_tol=cfg.cluster_tol)
    metrics.update({"steps": float(steps), "converged": float(converged)})
    return {"opinions": final_x, "metrics": metrics, "config": cfg, "graph": G}


def aggregate_records(records: List[Dict[str, object]], keys: Mapping[str, object]) -> Dict[str, object]:
    rows = [r["metrics"] for r in records]
    df = pd.DataFrame(rows)
    out: Dict[str, object] = dict(keys)
    for col in df.columns:
        out[col] = df[col].mean()
        out[f"{col}_std"] = df[col].std(ddof=1) if len(df) > 1 else 0.0
    return out


def save_dataframe(df: pd.DataFrame, name: str) -> Path:
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig: plt.Figure, name: str) -> Tuple[Path, Path]:
    """Save each formal figure as PNG, SVG, and PDF for draft/submission use."""
    png = FIG_DIR / f"{name}.png"
    svg = FIG_DIR / f"{name}.svg"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(svg)
    fig.savefig(pdf)
    return png, svg

def run_q1_baseline(mode: str = "preview", s: float = 0.5, alpha: float = 0.4) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Q1: same graph, same x0, same edge schedule; compare three mechanisms."""
    p = PRESETS[mode]
    n = int(p["n"])
    k = int(p["k"])
    max_steps = int(p["max_steps"])
    model_specs = [
        ("Classical DW", "classical", 0.0, 0.0),
        ("Symmetric biased DW", "symmetric", s, 0.0),
        ("Asymmetric biased DW", "asymmetric", s, alpha),
    ]
    rows: List[Dict[str, object]] = []
    exemplars: Dict[str, np.ndarray] = {}

    for rep in range(k):
        base_cfg = SimConfig(n=n, max_steps=max_steps, seed=42 + rep, model="asymmetric", s=s, alpha=alpha)
        G = make_graph(base_cfg)
        edges = graph_to_edges(G)
        x0 = initial_opinions(n, 1000 + rep)
        sched = edge_schedule(len(edges), max_steps, 2000 + rep)

        for label, model, s_val, a_val in model_specs:
            cfg = replace(base_cfg, model=model, s=s_val, alpha=a_val)
            result = simulate_once(cfg, G=G, x0=x0, schedule=sched)
            row = {"Question": "Q1", "rep": rep, "Model": label}
            row.update(result["metrics"])
            rows.append(row)
            if rep == 0:
                exemplars[label] = result["opinions"]

    df = pd.DataFrame(rows)
    save_dataframe(df, f"q1_baseline_{mode}")
    return df, exemplars




def plot_q1_baseline(df: pd.DataFrame, exemplars: Mapping[str, np.ndarray], mode: str = "preview") -> Tuple[plt.Figure, pd.DataFrame]:
    summary = df.groupby("Model", as_index=False).agg(
        P=("P", "mean"), P_std=("P", "std"),
        C=("C", "mean"), C_std=("C", "std"),
        D=("D", "mean"), D_std=("D", "std"),
        R_extreme=("R_extreme", "mean"),
        R_neutral=("R_neutral", "mean"),
        S_max_over_N=("S_max_over_N", "mean"),
    )
    order = ["Classical DW", "Symmetric biased DW", "Asymmetric biased DW"]
    short = {
        "Classical DW": "Classical",
        "Symmetric biased DW": "Symmetric",
        "Asymmetric biased DW": "Asymmetric",
    }
    palette = sns.color_palette("colorblind", n_colors=3)
    colors = dict(zip(order, palette))
    markers = {"Classical DW": "o", "Symmetric biased DW": "s", "Asymmetric biased DW": "^"}

    fig = plt.figure(figsize=(16.8, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.16, 1.0], hspace=0.54, wspace=0.44)
    ax_dist = fig.add_subplot(gs[0, :])

    for label in order:
        x = np.sort(np.asarray(exemplars[label], dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        ax_dist.plot(
            x, y, linewidth=2.8, marker=markers[label], markevery=max(1, len(x) // 12),
            markersize=8.5, label=short[label], color=colors[label], alpha=0.96,
        )
    ax_dist.axvspan(0.4, 0.6, color="0.86", alpha=0.32, zorder=0)
    ax_dist.axvline(0.5, color="0.32", linestyle=":", linewidth=1.5)
    ax_dist.set_xlim(0, 1)
    ax_dist.set_ylim(0, 1.01)
    ax_dist.set_xlabel(r"$x_i(T)$")
    ax_dist.set_ylabel(r"$F(x)$")
    ax_dist.legend(ncol=3, loc="upper left", frameon=True, handlelength=1.9, columnspacing=0.9)

    plot_df = df.copy()
    plot_df["Model label"] = plot_df["Model"].map(short)
    short_order = [short[x] for x in order]
    metrics = [("P", r"$P$"), ("C", r"$C$"), ("D", r"$D$")]
    for i, (metric, label_text) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, i])
        sns.barplot(
            data=plot_df, x="Model label", y=metric, hue="Model", order=short_order, hue_order=order,
            palette=colors, errorbar="sd", capsize=0.10, err_kws={"linewidth": 1.45},
            legend=False, ax=ax,
        )
        if metric == "D":
            ax.axhline(0, color="0.30", linewidth=1.35, linestyle=":")
        ax.set_xlabel("")
        ax.set_ylabel(label_text)
        ax.tick_params(axis="both", labelsize=20)
        ax.tick_params(axis="x", pad=13)
        for label in ax.get_xticklabels():
            label.set_rotation(18)
            label.set_ha("center")
            label.set_rotation_mode("anchor")
        ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.060, right=0.985, top=0.965, bottom=0.20)
    save_figure(fig, f"q1_baseline_{mode}")
    return fig, summary

def run_q2_phase_diagram(mode: str = "preview", s: float = 0.5, network: str = "ba") -> pd.DataFrame:
    """Q2: positive-parameter phase diagrams over d and alpha at fixed s."""
    p = PRESETS[mode]
    n = int(p["n"])
    k = int(p["k"])
    max_steps = int(p["max_steps"])
    d_values = list(p["d_values"])
    alpha_values = list(p["alpha_values"])
    rows: List[Dict[str, object]] = []

    for d_val in d_values:
        for alpha_val in alpha_values:
            records = []
            for rep in range(k):
                cfg = SimConfig(n=n, max_steps=max_steps, seed=3000 + rep, d=float(d_val), s=s, alpha=float(alpha_val), model="asymmetric", network=network)
                records.append(simulate_once(cfg, seed_offset=rep))
            rows.append(aggregate_records(records, {"Question": "Q2", "network": network.upper(), "s": s, "d": float(d_val), "alpha": float(alpha_val)}))

    df = pd.DataFrame(rows)
    save_dataframe(df, f"q2_phase_{network}_{mode}")
    return df




def heatmap_panel(
    df: pd.DataFrame,
    value_cols: Sequence[Tuple[str, str, str]],
    x: str,
    y: str,
    title: str,
    filename: str,
) -> plt.Figure:
    ncols = len(value_cols)
    fig, axes = plt.subplots(
        1, ncols, figsize=(5.15 * ncols, 5.05),
        gridspec_kw={"wspace": 0.36}
    )
    if ncols == 1:
        axes = [axes]

    for idx, (ax, (col, label, cmap)) in enumerate(zip(axes, value_cols)):
        mat = df.pivot(index=y, columns=x, values=col).sort_index(ascending=True)
        kwargs = dict(
            cmap=cmap,
            linewidths=0.14,
            linecolor="white",
            cbar_kws={"shrink": 0.78, "pad": 0.02},
        )
        if col == "D":
            vmax = max(0.05, float(np.nanmax(np.abs(mat.values))))
            kwargs["center"] = 0
            kwargs["vmin"] = -vmax
            kwargs["vmax"] = vmax
        sns.heatmap(mat, ax=ax, **kwargs)
        ax.set_title(label, pad=9)
        ax.set_xlabel(r"$d$" if x == "d" else x)
        ax.set_ylabel(
            r"$\alpha$" if y == "alpha" and idx == 0
            else (r"$p_{open}$" if y == "p_open" and idx == 0 else ""),
        )
        x_step = max(1, int(math.ceil(len(mat.columns) / 6)))
        y_step = max(1, int(math.ceil(len(mat.index) / 7)))
        ax.set_xticks(np.arange(len(mat.columns))[::x_step] + 0.5)
        ax.set_xticklabels([f"{float(v):.2g}" for v in mat.columns[::x_step]], rotation=0)
        ax.set_yticks(np.arange(len(mat.index))[::y_step] + 0.5)
        ax.set_yticklabels([f"{float(v):.2g}" for v in mat.index[::y_step]], rotation=0)
        ax.tick_params(axis="both", labelsize=20)
        if ax.collections and ax.collections[0].colorbar is not None:
            ax.collections[0].colorbar.ax.tick_params(labelsize=20)
    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.065, right=0.985)
    save_figure(fig, filename)
    return fig

def plot_q2_phase(df: pd.DataFrame, mode: str = "preview", network: str = "ba") -> plt.Figure:
    return heatmap_panel(
        df,
        [("P", "$P$", "viridis"), ("C", "$C$", "mako"), ("D", "$D$", "coolwarm"), ("R_extreme", "$R_{extreme}$", "rocket_r")],
        x="d",
        y="alpha",
        title="",
        filename=f"q2_phase_{network}_{mode}",
    )

def run_q3_open_minded(mode: str = "preview", s: float = 0.5, positive_alpha: float = 0.4) -> pd.DataFrame:
    """Q3: depolarization induced by open-minded agents with negative parameters."""
    p = PRESETS[mode]
    n = int(p["n"])
    k = int(p["k"])
    max_steps = int(p["max_steps"])
    rows: List[Dict[str, object]] = []

    for d_val in list(p["d_values"]):
        for p_open in list(p["p_open_values"]):
            records = []
            for rep in range(k):
                cfg = SimConfig(
                    n=n, max_steps=max_steps, seed=5000 + rep, d=float(d_val), s=s,
                    model="open_minded", p_open=float(p_open), positive_alpha_for_open=positive_alpha,
                )
                records.append(simulate_once(cfg, seed_offset=rep))
            rows.append(aggregate_records(records, {"Question": "Q3", "d": float(d_val), "p_open": float(p_open), "s": s}))

    df = pd.DataFrame(rows)
    save_dataframe(df, f"q3_open_minded_{mode}")
    return df


def plot_q3_open_minded(df: pd.DataFrame, mode: str = "preview") -> plt.Figure:
    return heatmap_panel(
        df,
        [("R_neutral", "$R_{neutral}$", "YlGnBu"), ("P", "$P$", "viridis"), ("R_extreme", "$R_{extreme}$", "rocket_r"), ("D", "$D$", "coolwarm")],
        x="d",
        y="p_open",
        title="",
        filename=f"q3_open_minded_{mode}",
    )

def run_q4_topology(mode: str = "preview", s: float = 0.5, alpha: float = 0.4, metric_grid: bool = True) -> pd.DataFrame:
    """Q4: compare ER/WS/BA while controlling N and average degree.

    The formal grid intentionally keeps three representative alpha values
    (-0.9, 0, 0.9), so each topology panel remains readable.
    """
    p = PRESETS[mode]
    n = int(p["n"])
    k = int(p["k"])
    max_steps = int(p["max_steps"])
    networks = ["er", "ws", "ba"]
    if metric_grid:
        d_values = list(p["d_values"])
        alpha_values = [-0.6, 0.0, 0.6] if mode == "preview" else [-0.9, 0.0, 0.9]
    else:
        d_values = [0.25]
        alpha_values = [alpha]

    rows: List[Dict[str, object]] = []
    for net in networks:
        for d_val in d_values:
            for alpha_val in alpha_values:
                records = []
                for rep in range(k):
                    cfg = SimConfig(
                        n=n, max_steps=max_steps, seed=7000 + rep, d=float(d_val), s=s,
                        alpha=float(alpha_val), model="asymmetric", network=net,
                    )
                    records.append(simulate_once(cfg, seed_offset=rep))
                rows.append(aggregate_records(records, {"Question": "Q4", "Network": net.upper(), "d": float(d_val), "alpha": float(alpha_val)}))

    df = pd.DataFrame(rows)
    save_dataframe(df, f"q4_topology_{mode}")
    return df



def plot_q4_topology(df: pd.DataFrame, mode: str = "preview", representative_d: float = 0.26, strong_alpha: float = 0.90) -> plt.Figure:
    """Readable topology figure with explicit numerical slices."""
    data = df.copy()
    data["Network"] = data["Network"].str.upper()
    network_order = ["ER", "WS", "BA"]
    colors = {"ER": "#0072B2", "WS": "#E69F00", "BA": "#009E73"}
    markers = {"ER": "o", "WS": "s", "BA": "^"}
    linestyles = {"ER": "-", "WS": "--", "BA": "-."}

    def nearest(values: pd.Series, target: float) -> float:
        arr = np.asarray(sorted(values.dropna().unique()), dtype=float)
        return float(arr[np.argmin(np.abs(arr - target))])

    def draw_panel(ax: plt.Axes, sub: pd.DataFrame, x: str, y: str, panel_title: str, xlabel: str, ylabel: str) -> None:
        for net in network_order:
            part = sub[sub["Network"] == net].sort_values(x)
            if part.empty:
                continue
            ax.plot(
                part[x], part[y], label=net, color=colors[net], marker=markers[net],
                linestyle=linestyles[net], linewidth=2.8, markersize=9.0, alpha=0.96,
            )
        ax.set_title(panel_title, pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, linewidth=0.55, alpha=0.45)

    d0 = nearest(data["d"], representative_d)
    alpha_hi = nearest(data["alpha"], strong_alpha)
    panel_a = data[np.isclose(data["d"].astype(float), d0)]
    panel_b = data[np.isclose(data["alpha"].astype(float), alpha_hi)]

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.9))
    draw_panel(
        axes[0], panel_a, "alpha", "D",
        rf"$D$ vs. $\alpha$ ($d={d0:.2f}$)", r"$\alpha$", r"$D$",
    )
    axes[0].axhline(0, color="0.30", linestyle=":", linewidth=1.45)
    draw_panel(
        axes[1], panel_b, "d", "P",
        rf"$P$ vs. $d$ ($\alpha={alpha_hi:.2f}$)", r"$d$", r"$P$",
    )
    draw_panel(
        axes[2], panel_b, "d", "C",
        rf"$C$ vs. $d$ ($\alpha={alpha_hi:.2f}$)", r"$d$", r"$C$",
    )

    axes[0].legend(title="Network", loc="upper right", frameon=True, handlelength=1.6, borderpad=0.35)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.86, wspace=0.34)
    save_figure(fig, f"q4_topology_{mode}")
    return fig

def run_q5_robustness(mode: str = "preview", s: float = 0.5, alpha: float = 0.4) -> pd.DataFrame:
    """Q5: finite-size, mu, cluster tolerance, Monte Carlo, and Tmax robustness."""
    p = PRESETS[mode]
    base_n = int(p["n"])
    k = int(p["k"])
    max_steps = int(p["max_steps"])
    n_values = list(p["n_values"])
    phases = {
        "High-confidence polarized": {"d": 0.45, "alpha": 0.0},
        "Directional polarized": {"d": 0.25, "alpha": alpha},
        "Fragmented": {"d": 0.06, "alpha": 0.0},
    }
    rows: List[Dict[str, object]] = []

    for phase, pars in phases.items():
        for n_val in n_values:
            cfg = SimConfig(n=int(n_val), max_steps=max_steps, seed=9001, d=pars["d"], s=s, alpha=pars["alpha"], model="asymmetric")
            result = simulate_once(cfg)
            row = {"Question": "Q5", "Test": "Finite size", "Phase": phase, "Setting": str(n_val), "SettingValue": float(n_val)}
            row.update(result["metrics"])
            rows.append(row)

        for mu_val in [0.1, 0.3, 0.5]:
            cfg = SimConfig(n=base_n, max_steps=max_steps, seed=9002, mu=mu_val, d=pars["d"], s=s, alpha=pars["alpha"], model="asymmetric")
            result = simulate_once(cfg)
            row = {"Question": "Q5", "Test": "Mu", "Phase": phase, "Setting": str(mu_val), "SettingValue": float(mu_val)}
            row.update(result["metrics"])
            rows.append(row)

    target_cfg = SimConfig(n=base_n, max_steps=max_steps, seed=9010, d=0.25, s=s, alpha=alpha, model="asymmetric")
    target = simulate_once(target_cfg)
    for ctol in [1e-2, 1e-3, 1e-4]:
        m = compute_metrics(target["opinions"], cluster_tol=ctol)
        row = {"Question": "Q5", "Test": "Cluster tolerance", "Phase": "Directional polarized", "Setting": f"{ctol:g}", "SettingValue": float(ctol)}
        row.update(m)
        rows.append(row)

    mc_seeds = list(range(max(5, k)))
    for count in [5, 10, 20, 30] if mode == "paper" else [2, 3, 5]:
        records = []
        for rep in mc_seeds[:min(count, len(mc_seeds))]:
            cfg = SimConfig(n=base_n, max_steps=max_steps, seed=9100 + rep, d=0.25, s=s, alpha=alpha, model="asymmetric")
            records.append(simulate_once(cfg))
        rows.append(aggregate_records(records, {"Question": "Q5", "Test": "Monte Carlo K", "Phase": "Directional polarized", "Setting": str(count), "SettingValue": float(count)}))

    for tmax in ([60_000, 120_000, 240_000] if mode == "preview" else [100_000, 300_000, 500_000]):
        cfg = SimConfig(n=base_n, max_steps=int(tmax), seed=9200, d=0.25, s=s, alpha=alpha, model="asymmetric")
        result = simulate_once(cfg)
        row = {"Question": "Q5", "Test": "Tmax", "Phase": "Directional polarized", "Setting": str(tmax), "SettingValue": float(tmax)}
        row.update(result["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    save_dataframe(df, f"q5_robustness_{mode}")
    return df


def plot_q5_robustness(df: pd.DataFrame, mode: str = "preview") -> plt.Figure:
    """Optional robustness figure; recommended for appendix/supplement if space is tight."""
    data = df.copy()
    data["Phase"] = data["Phase"].replace({
        "Consensus": "High-confidence polarized",
        "Polarization": "Directional polarized",
        "Fragmentation": "Fragmented",
    })
    tests = ["Finite size", "Mu", "Cluster tolerance", "Monte Carlo K", "Tmax"]
    xlabels = {
        "Finite size": r"System size $N$",
        "Mu": r"Convergence parameter $\mu$",
        "Cluster tolerance": r"Cluster tolerance $\epsilon_c$",
        "Monte Carlo K": r"Monte Carlo count $K$",
        "Tmax": r"Maximum time $T_{max}$",
    }
    phase_order = ["High-confidence polarized", "Directional polarized", "Fragmented"]
    colors = {
        "High-confidence polarized": "#0072B2",
        "Directional polarized": "#D55E00",
        "Fragmented": "#009E73",
    }
    markers = {"High-confidence polarized": "o", "Directional polarized": "s", "Fragmented": "^"}
    linestyles = {"High-confidence polarized": "-", "Directional polarized": "--", "Fragmented": "-."}

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.8))
    axes = axes.ravel()
    for ax, test in zip(axes, tests):
        sub = data[data["Test"] == test].copy()
        metric = "C" if test == "Cluster tolerance" else "P"
        for phase in phase_order:
            part = sub[sub["Phase"] == phase].sort_values("SettingValue")
            if part.empty:
                continue
            ax.plot(
                part["SettingValue"], part[metric], label=phase,
                color=colors[phase], marker=markers[phase], linestyle=linestyles[phase],
                linewidth=2.8, markersize=9.0,
            )
        ax.set_title(f"{test}: {metric}", pad=8)
        ax.set_xlabel(xlabels[test])
        ax.set_ylabel(metric)
        ax.grid(True, linewidth=0.55, alpha=0.45)
        if test in {"Cluster tolerance", "Tmax"}:
            ax.set_xscale("log")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title="Representative regime", loc="center", frameon=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.92, hspace=0.46, wspace=0.34)
    save_figure(fig, f"q5_robustness_optional_appendix_{mode}")
    return fig

def run_all(mode: str = "preview") -> Dict[str, pd.DataFrame]:
    """Convenience runner for all five questions."""
    outputs: Dict[str, pd.DataFrame] = {}
    q1, q1_ex = run_q1_baseline(mode=mode)
    plot_q1_baseline(q1, q1_ex, mode=mode)
    outputs["q1"] = q1

    q2 = run_q2_phase_diagram(mode=mode)
    plot_q2_phase(q2, mode=mode)
    outputs["q2"] = q2

    q3 = run_q3_open_minded(mode=mode)
    plot_q3_open_minded(q3, mode=mode)
    outputs["q3"] = q3

    q4 = run_q4_topology(mode=mode)
    plot_q4_topology(q4, mode=mode)
    outputs["q4"] = q4

    q5 = run_q5_robustness(mode=mode)
    plot_q5_robustness(q5, mode=mode)
    outputs["q5"] = q5
    return outputs