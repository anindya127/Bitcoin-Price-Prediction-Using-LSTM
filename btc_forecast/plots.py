"""Figures for the README."""
from __future__ import annotations

import numpy as np


def plot_last_fold(result: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = result["_plot"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(p["dates"], p["actual"], color="black", lw=1.5, label="Actual close")
    for name, ret in p["preds"].items():
        if name.startswith(("Drift", "Moving")):
            continue
        ax.plot(p["dates"], p["prev"] * np.exp(ret), lw=1, alpha=0.8, label=name)
    ax.set_ylabel("BTC-USD close (USD)")
    ax.set_title("Last walk-forward fold: next-day price forecasts")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ratio_bars(rows: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["model"] for r in rows]
    means = [r["rmse_ratio_vs_naive"][0] for r in rows]
    errs = [r["rmse_ratio_vs_naive"][1] for r in rows]
    ax.bar(range(len(rows)), means, yerr=errs, capsize=4, color="#4c78a8")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="Naive = 1.0")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    lo = min(means) - max(errs) - 0.01
    ax.set_ylim(min(0.98, lo), max(means) + max(errs) + 0.02)
    ax.set_ylabel("RMSE ratio vs naive (lower is better)")
    ax.set_title("Price-level RMSE relative to the naive forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
