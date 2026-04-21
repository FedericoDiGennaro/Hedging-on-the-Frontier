#!/usr/bin/env python3
"""
Generate compact transfer-summary `.npz` files for the K-comparison plots.

These summary files are already bundled under:

    final_scripts/raw_data/transfer/

This script is provided so another user can regenerate them from a full
`experiments/adaptive_pruning/results/` directory that contains the cached
`*_data_cache.pkl` files from the original experiments.

Example:
    python final_scripts/transfer/prepare_transfer_data.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT.parent / "experiments" / "adaptive_pruning" / "results"
DEFAULT_OUTPUT_ROOT = ROOT / "raw_data" / "transfer"

HELM_KS = [1, 2, 3, 4, 5, 6]
VHELM_KS = [1, 2, 3, 4, 5, 6, 7, 8]


def install_dummy_tqdm() -> None:
    """Allow importing the experiment code without the optional tqdm dependency."""
    if "tqdm" in sys.modules:
        return
    module = types.ModuleType("tqdm")
    module.tqdm = lambda iterable, *args, **kwargs: iterable
    sys.modules["tqdm"] = module


def import_run_single_combination():
    install_dummy_tqdm()
    adaptive_pruning_dir = ROOT.parent / "experiments" / "adaptive_pruning"
    sys.path.insert(0, str(adaptive_pruning_dir))
    from run_experiment import run_single_combination  # type: ignore

    return run_single_combination


def compute_robust_baseline(source_losses, target_losses) -> float:
    source_risks = np.array([np.mean(src, axis=1) for src in source_losses]).T
    robust_idx = int(np.argmin(np.max(source_risks, axis=1)))
    target_risks = np.mean(target_losses, axis=1)
    return float(target_risks[robust_idx] - np.min(target_risks))


def summarize_dataset(cache_path: Path, run_single_combination) -> dict[str, np.ndarray]:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    all_source_losses = cache["all_source_losses"]
    all_target_losses = cache["all_target_losses"]
    n_grid = cache["n_grid"]

    n_combos = len(all_source_losses)
    n_sizes = len(n_grid)
    mean_all_per_combo = np.zeros((n_combos, n_sizes))
    mean_adapt_per_combo = np.zeros((n_combos, n_sizes))
    mean_pareto_per_combo = np.zeros((n_combos, n_sizes))
    robust_per_combo = np.zeros(n_combos)

    for idx in range(n_combos):
        src_losses = all_source_losses[idx]
        tgt_losses = all_target_losses[idx]
        excess_risk_all, excess_risk_adapt, excess_risk_pareto, _, _, _ = run_single_combination(
            src_losses,
            tgt_losses,
            n_grid,
            B=500,
        )
        mean_all_per_combo[idx] = np.nanmean(excess_risk_all, axis=1)
        mean_adapt_per_combo[idx] = np.nanmean(excess_risk_adapt, axis=1)
        mean_pareto_per_combo[idx] = np.nanmean(excess_risk_pareto, axis=1)
        robust_per_combo[idx] = compute_robust_baseline(src_losses, tgt_losses)

    mean_all = np.nanmean(mean_all_per_combo, axis=0)
    mean_adapt = np.nanmean(mean_adapt_per_combo, axis=0)
    mean_pareto = np.nanmean(mean_pareto_per_combo, axis=0)
    mean_adapt[0] = mean_pareto[0]

    return {
        "n_grid": n_grid,
        "mean_all": mean_all,
        "mean_adapt": mean_adapt,
        "mean_pareto": mean_pareto,
        "mean_robust": np.array(np.nanmean(robust_per_combo)),
        "q25_all": np.nanpercentile(mean_all_per_combo, 25, axis=0),
        "q75_all": np.nanpercentile(mean_all_per_combo, 75, axis=0),
        "q25_adapt": np.nanpercentile(mean_adapt_per_combo, 25, axis=0),
        "q75_adapt": np.nanpercentile(mean_adapt_per_combo, 75, axis=0),
        "q25_pareto": np.nanpercentile(mean_pareto_per_combo, 25, axis=0),
        "q75_pareto": np.nanpercentile(mean_pareto_per_combo, 75, axis=0),
        "q25_robust": np.array(np.nanpercentile(robust_per_combo, 25)),
        "q75_robust": np.array(np.nanpercentile(robust_per_combo, 75)),
    }


def write_summary_files(
    source_root: Path,
    output_root: Path,
    dataset_name: str,
    ks: list[int],
    run_single_combination,
) -> None:
    dataset_prefix = dataset_name.lower()
    for k in ks:
        cache_path = source_root / dataset_name / f"K{k}" / f"{dataset_prefix}_data_cache.pkl"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing cache for {dataset_name} K={k}: {cache_path}"
            )
        summary = summarize_dataset(cache_path, run_single_combination)
        out_dir = output_root / dataset_prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"k{k}_summary.npz"
        np.savez(out_path, **summary)
        print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact transfer-summary `.npz` files for the K-comparison plots."
    )
    parser.add_argument(
        "--dataset",
        choices=["helm", "vhelm", "all"],
        default="all",
        help="Which dataset summaries to regenerate.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Path to experiments/adaptive_pruning/results from the full repo.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where compact summary `.npz` files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_single_combination = import_run_single_combination()

    if args.dataset in {"helm", "all"}:
        write_summary_files(
            source_root=args.source_root,
            output_root=args.output_root,
            dataset_name="HELM",
            ks=HELM_KS,
            run_single_combination=run_single_combination,
        )

    if args.dataset in {"vhelm", "all"}:
        write_summary_files(
            source_root=args.source_root,
            output_root=args.output_root,
            dataset_name="VHELM",
            ks=VHELM_KS,
            run_single_combination=run_single_combination,
        )


if __name__ == "__main__":
    main()
