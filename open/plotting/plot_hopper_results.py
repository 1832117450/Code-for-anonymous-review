#!/usr/bin/env python3
"""Plot public hopper-medium-v2 CSV results.

This script reads only the CSV files under ``results/csv`` and produces:

1. dense and sparse learning curves with mean and half-std bands over seeds;
2. a final-score bar plot computed from the last 10 evaluation points.

It is intentionally small and independent of private datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["BCQ", "TD3+BC", "IQL", "CQL", "TD3+NC", "Ours"]
COLORS = {
    "BCQ": "#8B5CF6",
    "TD3+BC": "#10B981",
    "IQL": "#F59E0B",
    "CQL": "#EF4444",
    "TD3+NC": "#6366F1",
    "Ours": "#2563EB",
}
LINESTYLES = {
    "BCQ": "-",
    "TD3+BC": "-",
    "IQL": "-",
    "CQL": "-",
    "TD3+NC": "--",
    "Ours": "-",
}


def _read_seed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "step" not in df.columns or "norm_score" not in df.columns:
        raise ValueError(f"CSV must contain step and norm_score columns: {path}")
    return df[["step", "norm_score"]].copy()


def _collect_group(csv_files: list[Path]) -> pd.DataFrame:
    frames = []
    for seed, path in enumerate(sorted(csv_files)):
        df = _read_seed_csv(path)
        df["seed_index"] = seed
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["step", "mean", "std"])
    all_df = pd.concat(frames, ignore_index=True)
    return (
        all_df.groupby("step", as_index=False)["norm_score"]
        .agg(mean="mean", std="std")
        .fillna({"std": 0.0})
        .sort_values("step")
    )


def result_groups(csv_root: Path) -> dict[str, dict[str, list[Path]]]:
    return {
        "dense": {
            "BCQ": list((csv_root / "baselines/BCQ").glob("*.csv")),
            "TD3+BC": list((csv_root / "baselines/TD3_BC").glob("*.csv")),
            "IQL": list((csv_root / "baselines/IQL").glob("*.csv")),
            "CQL": list((csv_root / "baselines/CQL").glob("*.csv")),
            "TD3+NC": list((csv_root / "methods/dense/td3_nc").glob("*.csv")),
        },
        "sparse": {
            "BCQ": list((csv_root / "baselines/BCQ_Sparse").glob("*.csv")),
            "TD3+BC": list((csv_root / "baselines/TD3_BC_Sparse").glob("*.csv")),
            "IQL": list((csv_root / "baselines/IQL_Sparse").glob("*.csv")),
            "CQL": list((csv_root / "baselines/CQL_Sparse").glob("*.csv")),
            "TD3+NC": list((csv_root / "methods/sparse/td3_nc").glob("*.csv")),
            "Ours": list((csv_root / "methods/sparse/ours").glob("*.csv")),
        },
    }


def load_results(csv_root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    groups = result_groups(csv_root)
    return {
        reward: {method: _collect_group(paths) for method, paths in method_map.items()}
        for reward, method_map in groups.items()
    }


def final_scores(csv_root: Path, last_n: int) -> pd.DataFrame:
    records = []
    for reward, method_map in result_groups(csv_root).items():
        for method, paths in method_map.items():
            seed_scores = []
            for path in sorted(paths):
                df = _read_seed_csv(path)
                if df.empty:
                    continue
                seed_scores.append(float(df["norm_score"].tail(last_n).mean()))
            if not seed_scores:
                continue
            records.append(
                {
                    "reward": reward,
                    "method": method,
                    "seeds": len(seed_scores),
                    "last10_mean": float(np.mean(seed_scores)),
                    "last10_std": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
                }
            )
    return pd.DataFrame(records)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_learning_curves(curves: dict[str, dict[str, pd.DataFrame]], out_path: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=False)
    for ax, reward, title in zip(axes, ["dense", "sparse"], ["Dense reward", "Sparse delayed reward"]):
        for method in METHOD_ORDER:
            curve = curves.get(reward, {}).get(method)
            if curve is None or curve.empty:
                continue
            x = curve["step"].to_numpy() / 1000.0
            y = curve["mean"].to_numpy()
            std = 0.5 * curve["std"].to_numpy()
            ax.plot(
                x,
                y,
                label=method,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.5,
            )
            ax.fill_between(x, y - std, y + std, color=COLORS[method], alpha=0.10, linewidth=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"Training Steps ($\times 10^3$)")
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Normalized Score")
    axes[1].legend(loc="lower right", ncol=2, frameon=True, framealpha=0.9)
    fig.tight_layout(w_pad=1.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_final_bars(summary: pd.DataFrame, out_path: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, reward, title in zip(axes, ["dense", "sparse"], ["Dense reward", "Sparse delayed reward"]):
        sub = summary[summary["reward"] == reward].copy()
        sub["order"] = sub["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
        sub = sub.sort_values("order")
        x = np.arange(len(sub))
        colors = [COLORS[m] for m in sub["method"]]
        ax.bar(x, sub["last10_mean"], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["method"], rotation=30, ha="right")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Last-10 Normalized Score")
    fig.tight_layout(w_pad=1.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot public hopper-medium-v2 results.")
    parser.add_argument("--csv-root", type=Path, default=Path("results/csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--last-n", type=int, default=10)
    args = parser.parse_args()

    curves = load_results(args.csv_root)
    summary = final_scores(args.csv_root, args.last_n)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "hopper_last10_summary.csv", index=False)
    plot_learning_curves(curves, args.out_dir / "hopper_learning_curves.pdf")
    plot_final_bars(summary, args.out_dir / "hopper_last10_bars.pdf")
    print(f"Saved plots and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
