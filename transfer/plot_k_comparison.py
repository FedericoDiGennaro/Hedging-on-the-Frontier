#!/usr/bin/env python3
"""
Create the transfer K-comparison plots from bundled summary `.npz` files.

Inputs:
    final_scripts/raw_data/transfer/helm/k*_summary.npz
    final_scripts/raw_data/transfer/vhelm/k*_summary.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = ROOT / "raw_data" / "transfer"
OUTPUT_DIR = ROOT / "plots" / "transfer"

HELM_KS = [1, 2, 3, 4, 5, 6]
VHELM_KS = [1, 2, 3, 4, 5, 6, 7, 8]


def load_summary(summary_root: Path, dataset: str, k: int) -> dict[str, np.ndarray] | None:
    path = summary_root / dataset / f"k{k}_summary.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def plot_comparison_subplots(results_by_k: dict[int, dict[str, np.ndarray]], dataset_name: str, output_path: Path) -> None:
    if not results_by_k:
        print(f"No results available for {dataset_name}; skipping.")
        return

    ks = sorted(results_by_k)
    n_plots = len(ks)
    n_cols = 3 if n_plots <= 6 else 4
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.0),
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    handles = None
    labels = None

    for idx, k in enumerate(ks):
        res = results_by_k[k]
        ax = axes[idx]

        def err(center_key: str, low_key: str, high_key: str):
            center = res[center_key]
            lower = res[low_key]
            upper = res[high_key]
            return [center - lower, upper - center]

        ax.errorbar(
            res["n_grid"],
            res["mean_all"],
            yerr=err("mean_all", "q25_all", "q75_all"),
            fmt="o-",
            color="C0",
            linewidth=1.8,
            markersize=5,
            alpha=0.9,
            elinewidth=1.3,
            capsize=3,
            label="ERM-all",
        )
        ax.errorbar(
            res["n_grid"],
            res["mean_adapt"],
            yerr=err("mean_adapt", "q25_adapt", "q75_adapt"),
            fmt="s-",
            color="C1",
            linewidth=1.8,
            markersize=5,
            alpha=0.9,
            elinewidth=1.3,
            capsize=3,
            label="ERM-adaptive",
        )
        ax.errorbar(
            res["n_grid"],
            res["mean_pareto"],
            yerr=err("mean_pareto", "q25_pareto", "q75_pareto"),
            fmt="^-",
            color="C2",
            linewidth=1.8,
            markersize=5,
            alpha=0.9,
            elinewidth=1.3,
            capsize=3,
            label="ERM-Pareto",
        )

        mean_robust = float(res["mean_robust"])
        q25_robust = float(res["q25_robust"])
        q75_robust = float(res["q75_robust"])
        ax.axhline(mean_robust, color="C3", linestyle="-", linewidth=2.0, alpha=0.9, label="minmax")
        ax.errorbar(
            [res["n_grid"][0]],
            [mean_robust],
            yerr=[[mean_robust - q25_robust], [q75_robust - mean_robust]],
            fmt="none",
            ecolor="C3",
            elinewidth=1.3,
            capsize=3,
        )

        ax.set_title(f"{dataset_name} (K={k})", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel("Mean Excess Risk", fontsize=10)
        if idx >= (n_rows - 1) * n_cols or idx == n_plots - 1:
            ax.set_xlabel("Target sample size (n)", fontsize=10)

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    if handles is not None and labels is not None:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {output_path}")
    print(f"wrote {output_path.with_suffix('.pdf')}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the transfer K-comparison plots from bundled summary `.npz` files."
    )
    parser.add_argument(
        "--dataset",
        choices=["helm", "vhelm", "all"],
        default="all",
        help="Which K-comparison plot(s) to generate.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=RAW_DATA_ROOT,
        help="Root directory containing bundled transfer summary `.npz` files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the generated figures will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset in {"helm", "all"}:
        helm_results = {
            k: summary
            for k in HELM_KS
            if (summary := load_summary(args.summary_root, "helm", k)) is not None
        }
        plot_comparison_subplots(helm_results, "HELM", args.output_dir / "helm_k_comparison.png")

    if args.dataset in {"vhelm", "all"}:
        vhelm_results = {
            k: summary
            for k in VHELM_KS
            if (summary := load_summary(args.summary_root, "vhelm", k)) is not None
        }
        plot_comparison_subplots(vhelm_results, "VHELM", args.output_dir / "vhelm_k_comparison.png")


if __name__ == "__main__":
    main()
