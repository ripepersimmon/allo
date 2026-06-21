from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Callable

from src.portfolio import gmv_long_only, gmv_unconstrained, effective_n, turnover

CRISIS_PERIODS = {
    "GFC":    ("2007-01-01", "2009-06-30"),
    "COVID":  ("2019-10-01", "2020-09-30"),
    "Rates":  ("2021-07-01", "2023-01-31"),
}

ESTIMATOR_COLORS = {
    "Sample": "#e41a1c",
    "LW":     "#377eb8",
    "Gerber": "#4daf4a",
}


def rolling_gmv(
    returns: pd.DataFrame,
    estimator_fn: Callable,
    window: int = 252,
    constrained: bool = True,
) -> pd.DataFrame:
    """Roll a window over returns, estimate cov, solve GMV. Returns weight DataFrame.

    Tickers with any NaN in a window (not yet listed / data gap) are excluded
    from that window's optimisation; their weight is set to 0.
    """
    tickers = returns.columns.tolist()
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)
    solver = gmv_long_only if constrained else gmv_unconstrained
    rows = {}
    dates = returns.index

    for i in range(window, len(dates)):
        win = returns.iloc[i - window : i].dropna(axis=1)
        if win.shape[1] < 2:
            continue
        w_full = np.zeros(n)
        try:
            cov = estimator_fn(win)
            w_active = solver(cov)
        except Exception:
            w_active = np.full(win.shape[1], 1.0 / win.shape[1])
        np.put(w_full, [ticker_idx[t] for t in win.columns], w_active)
        rows[dates[i]] = w_full

    df = pd.DataFrame(rows, index=tickers).T
    df.index = pd.to_datetime(df.index)
    return df


def compute_metrics(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Return effective_n and turnover time-series for a weight DataFrame."""
    eff_n = weights_df.apply(effective_n, axis=1)
    to_series = pd.Series(
        [np.nan]
        + [turnover(weights_df.iloc[i - 1].values, weights_df.iloc[i].values)
           for i in range(1, len(weights_df))],
        index=weights_df.index,
    )
    return pd.DataFrame({"effective_n": eff_n, "turnover": to_series})


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_weights(
    weights_dict: dict[str, pd.DataFrame],
    crisis_name: str,
    save_path: str | None = None,
) -> None:
    start, end = CRISIS_PERIODS[crisis_name]
    fig, axes = plt.subplots(len(weights_dict), 1, figsize=(14, 4 * len(weights_dict)), sharex=True)
    if len(weights_dict) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, weights_dict.items()):
        sub = df.loc[start:end]
        sub.plot.area(ax=ax, stacked=True, legend=False, linewidth=0)
        ax.set_title(f"{name} — {crisis_name}", fontsize=12)
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.13, 0.98), fontsize=8)
    fig.suptitle(f"GMV Weights — {crisis_name} Crisis", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_concentration(
    weights_dict: dict[str, pd.DataFrame],
    crisis_name: str,
    save_path: str | None = None,
) -> None:
    start, end = CRISIS_PERIODS[crisis_name]
    fig, ax = plt.subplots(figsize=(13, 4))

    for name, df in weights_dict.items():
        sub = df.loc[start:end]
        eff = sub.apply(effective_n, axis=1)
        ax.plot(eff.index, eff.values, label=name, color=ESTIMATOR_COLORS.get(name), linewidth=1.8)

    ax.set_title(f"Effective N (1/HHI) — {crisis_name} Crisis")
    ax.set_ylabel("Effective N")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cov_heatmap(cov: np.ndarray, labels: list[str], title: str, save_path: str | None = None) -> None:
    # Convert to correlation for readability
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=labels, yticklabels=labels,
        vmin=-1, vmax=1, linewidths=0.3, ax=ax,
    )
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_turnover(
    weights_dict: dict[str, pd.DataFrame],
    crisis_name: str,
    save_path: str | None = None,
) -> None:
    start, end = CRISIS_PERIODS[crisis_name]
    avg_to = {}
    for name, df in weights_dict.items():
        sub = df.loc[start:end]
        to_vals = [
            turnover(sub.iloc[i - 1].values, sub.iloc[i].values)
            for i in range(1, len(sub))
        ]
        avg_to[name] = np.mean(to_vals) if to_vals else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [ESTIMATOR_COLORS.get(n, "gray") for n in avg_to]
    ax.bar(list(avg_to.keys()), list(avg_to.values()), color=colors)
    ax.set_title(f"Average Daily Turnover — {crisis_name} Crisis")
    ax.set_ylabel("Turnover")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def summary_table(weights_dict: dict[str, pd.DataFrame], crisis_name: str) -> pd.DataFrame:
    start, end = CRISIS_PERIODS[crisis_name]
    rows = []
    for name, df in weights_dict.items():
        sub = df.loc[start:end]
        to_vals = [
            turnover(sub.iloc[i - 1].values, sub.iloc[i].values)
            for i in range(1, len(sub))
        ]
        rows.append({
            "Estimator": name,
            "Avg Effective N": sub.apply(effective_n, axis=1).mean(),
            "Min Effective N": sub.apply(effective_n, axis=1).min(),
            "Max Single Weight": sub.max().max(),
            "Avg Turnover": np.mean(to_vals) if to_vals else np.nan,
        })
    return pd.DataFrame(rows).set_index("Estimator").round(4)
