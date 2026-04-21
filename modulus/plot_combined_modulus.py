#!/usr/bin/env python3
"""
Create the final combined upper/lower modulus figure from the bundled HELM/VHELM `.npz` files.

The exact cache files used for the paper-style combined plots are included under:

    final_scripts/raw_data/modulus/helm/
    final_scripts/raw_data/modulus/vhelm/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "raw_data" / "modulus"
OUTPUT_DIR = ROOT / "plots" / "modulus"


def summary_with_iqr(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.median(array, axis=0),
        np.percentile(array, 25, axis=0),
        np.percentile(array, 75, axis=0),
    )


def load_dataset(cache_root: Path, dataset_name: str, gamma: float) -> np.lib.npyio.NpzFile:
    npz_path = cache_root / dataset_name / f"modulus_curves_gamma{gamma:.3f}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing cached data: {npz_path}\n"
            f"Make sure the bundled modulus `.npz` files are present under "
            f"`final_scripts/raw_data/modulus/{dataset_name}/`."
        )
    return np.load(npz_path)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def plot_upper_and_lower_linear_cropped(
    cache_root: Path,
    output_dir: Path,
    gamma: float,
    lower_xlim: tuple[float, float] = (0.0, 0.5),
) -> None:
    helm = load_dataset(cache_root, "helm", gamma)
    vhelm = load_dataset(cache_root, "vhelm", gamma)

    t_helm = helm["t_grid"]
    t_vhelm = vhelm["t_grid"]

    upper_med_helm, upper_q25_helm, upper_q75_helm = summary_with_iqr(helm["v_upper_q99"])
    upper_med_vhelm, upper_q25_vhelm, upper_q75_vhelm = summary_with_iqr(vhelm["v_upper_q99"])
    lower_med_helm, lower_q25_helm, lower_q75_helm = summary_with_iqr(helm["v_lower_q01"])
    lower_med_vhelm, lower_q25_vhelm, lower_q75_vhelm = summary_with_iqr(vhelm["v_lower_q01"])
    pi_median_vhelm = np.median(vhelm["pi_upper"], axis=0)

    valid_vhelm = (t_vhelm > 0) & (pi_median_vhelm > 0)
    valid_helm = t_helm > 0

    t_lower_vhelm = t_vhelm[valid_vhelm]
    lower_med_vhelm = lower_med_vhelm[valid_vhelm]
    lower_q25_vhelm = lower_q25_vhelm[valid_vhelm]
    lower_q75_vhelm = lower_q75_vhelm[valid_vhelm]

    t_lower_helm = t_helm[valid_helm]
    lower_med_helm = lower_med_helm[valid_helm]
    lower_q25_helm = lower_q25_helm[valid_helm]
    lower_q75_helm = lower_q75_helm[valid_helm]

    def trim_decrease(
        t_values: np.ndarray,
        median_values: np.ndarray,
        q25_values: np.ndarray,
        q75_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        peak_idx = len(median_values) - 1
        for idx in range(1, len(median_values)):
            # Ignore tiny non-monotonic wiggles while the curve is still negative.
            if median_values[idx - 1] <= 0:
                continue
            decrease = median_values[idx - 1] - median_values[idx]
            if decrease > 0.01:
                peak_idx = idx - 1
                break
            previous = abs(median_values[idx - 1])
            if previous > 0 and decrease / previous > 0.01:
                peak_idx = idx - 1
                break
        return (
            t_values[: peak_idx + 1],
            median_values[: peak_idx + 1],
            q25_values[: peak_idx + 1],
            q75_values[: peak_idx + 1],
        )

    t_lower_helm, lower_med_helm, lower_q25_helm, lower_q75_helm = trim_decrease(
        t_lower_helm, lower_med_helm, lower_q25_helm, lower_q75_helm
    )
    t_lower_vhelm, lower_med_vhelm, lower_q25_vhelm, lower_q75_vhelm = trim_decrease(
        t_lower_vhelm, lower_med_vhelm, lower_q25_vhelm, lower_q75_vhelm
    )

    def zero_crossing(t_values: np.ndarray, y_values: np.ndarray) -> float | None:
        for idx in range(len(y_values) - 1):
            if y_values[idx] < 0 <= y_values[idx + 1]:
                t1, t2 = t_values[idx], t_values[idx + 1]
                y1, y2 = y_values[idx], y_values[idx + 1]
                return float(t1 + (0 - y1) * (t2 - t1) / (y2 - y1))
        return None

    crossing_helm = zero_crossing(t_lower_helm, lower_med_helm)
    crossing_vhelm = zero_crossing(t_lower_vhelm, lower_med_vhelm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.7))

    ax1.fill_between(t_helm, upper_q25_helm, upper_q75_helm, color="steelblue", alpha=0.15, linewidth=0)
    ax1.fill_between(t_vhelm, upper_q25_vhelm, upper_q75_vhelm, color="coral", alpha=0.15, linewidth=0)
    ax1.plot(t_helm, upper_med_helm, "-", color="steelblue", linewidth=2.5)
    ax1.plot(t_vhelm, upper_med_vhelm, "-", color="coral", linewidth=2.5)
    ax1.set_xlabel(r"$t$", fontsize=13)
    ax1.set_title("Upper Modulus", fontsize=15)
    ax1.set_ylim(0, None)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax1.tick_params(axis="both", which="major", labelsize=10)

    ax2.fill_between(t_lower_helm, lower_q25_helm, lower_q75_helm, color="steelblue", alpha=0.15, linewidth=0)
    ax2.fill_between(t_lower_vhelm, lower_q25_vhelm, lower_q75_vhelm, color="coral", alpha=0.15, linewidth=0)
    ax2.plot(t_lower_helm, lower_med_helm, "-", color="steelblue", linewidth=2.5)
    ax2.plot(t_lower_vhelm, lower_med_vhelm, "-", color="coral", linewidth=2.5)
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=1.5, alpha=0.7, zorder=2)
    if crossing_helm is not None:
        ax2.axvline(x=crossing_helm, color="steelblue", linestyle=":", linewidth=1.5, alpha=0.4, zorder=2)
    if crossing_vhelm is not None:
        ax2.axvline(x=crossing_vhelm, color="coral", linestyle=":", linewidth=1.5, alpha=0.4, zorder=2)
    ax2.set_xlabel(r"$t$", fontsize=13)
    ax2.set_title("Lower Modulus", fontsize=15)
    ax2.set_xlim(*lower_xlim)
    y_lo = min(lower_q25_helm.min(), lower_q25_vhelm.min(), lower_med_helm.min(), lower_med_vhelm.min())
    y_hi = max(lower_q75_helm.max(), lower_q75_vhelm.max(), lower_med_helm.max(), lower_med_vhelm.max())
    padding = 0.05 * max(1e-9, y_hi - y_lo)
    ax2.set_ylim(y_lo - padding, y_hi + padding)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax2.tick_params(axis="both", which="major", labelsize=10)

    handles = [
        Line2D([0], [0], color="steelblue", linewidth=2.5, label="HELM"),
        Line2D([0], [0], color="coral", linewidth=2.5, label="VHELM"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=12, frameon=False)

    plt.tight_layout()
    save_figure(fig, output_dir, "combined_upper_and_lower_modulus_linear_cropped")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the final combined upper/lower modulus figure from cached `.npz` files."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="Margin value / cache filename suffix to plot.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root directory containing the cached `helm/` and `vhelm/` modulus `.npz` files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the figures will be written.",
    )
    parser.add_argument(
        "--plot",
        choices=["upper_lower"],
        default="upper_lower",
        help="Which combined figure to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_upper_and_lower_linear_cropped(args.cache_root, args.output_dir, args.gamma)


if __name__ == "__main__":
    main()
