#!/usr/bin/env python3
"""
Download the raw HELM and VHELM benchmark directories used by the minimal
`final_scripts` experiments.

Expected destination layout:

    final_scripts/raw_data/
      helm/benchmark_output/runs/v1.0.0/
      vhelm/benchmark_output/runs/v2.0.0/

Notes
-----
- HELM defaults to the public HELM Lite bucket already referenced in this repo.
- VHELM requires an explicit source path unless the environment variable
  `VHELM_BENCHMARK_OUTPUT_PATH` is set.
- A source path can be either:
  1. a GCS bucket/prefix (e.g. `gs://...`) and will be downloaded with
     `gcloud storage rsync -r`, or
  2. a local directory that contains `runs/<suite_version>`, in which case the
     files are copied locally.

Examples
--------
Dry run for both datasets:
    python final_scripts/raw_data/download_data.py --dry-run

Download HELM only:
    python final_scripts/raw_data/download_data.py --dataset helm

Download VHELM from a provided source:
    python final_scripts/raw_data/download_data.py \
        --dataset vhelm \
        --vhelm-source gs://YOUR_BUCKET/benchmark_output
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    suite_version: str
    source_root: str | None
    dest_root: Path

    @property
    def source_runs_path(self) -> str | Path:
        if not self.source_root:
            raise ValueError(f"No source configured for dataset '{self.name}'.")
        source_root = self.source_root.rstrip("/")
        if source_root.startswith("gs://"):
            return f"{source_root}/runs/{self.suite_version}"
        return Path(source_root) / "runs" / self.suite_version

    @property
    def dest_runs_path(self) -> Path:
        return self.dest_root / "benchmark_output" / "runs" / self.suite_version


def build_specs(args: argparse.Namespace) -> dict[str, DatasetSpec]:
    helm_source = args.helm_source or os.environ.get(
        "HELM_BENCHMARK_OUTPUT_PATH",
        "gs://crfm-helm-public/lite/benchmark_output",
    )
    vhelm_source = args.vhelm_source or os.environ.get("VHELM_BENCHMARK_OUTPUT_PATH")

    return {
        "helm": DatasetSpec(
            name="helm",
            suite_version=args.helm_version,
            source_root=helm_source,
            dest_root=THIS_DIR / "helm",
        ),
        "vhelm": DatasetSpec(
            name="vhelm",
            suite_version=args.vhelm_version,
            source_root=vhelm_source,
            dest_root=THIS_DIR / "vhelm",
        ),
    }


def ensure_dest_parent(spec: DatasetSpec) -> None:
    spec.dest_runs_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dataset_root(spec: DatasetSpec) -> None:
    spec.dest_root.mkdir(parents=True, exist_ok=True)


def download_from_gcs(spec: DatasetSpec, dry_run: bool) -> None:
    if shutil.which("gcloud") is None:
        raise RuntimeError(
            "gcloud is not installed or not on PATH. Install the Google Cloud CLI "
            "or use a local source path instead."
        )

    cmd = [
        "gcloud",
        "storage",
        "rsync",
        "-r",
        str(spec.source_runs_path),
        str(spec.dest_runs_path),
    ]
    print(f"[{spec.name}] {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def copy_from_local(spec: DatasetSpec, dry_run: bool) -> None:
    source_path = Path(spec.source_runs_path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Local source path does not exist for {spec.name}: {source_path}"
        )

    print(f"[{spec.name}] copy {source_path} -> {spec.dest_runs_path}")
    if not dry_run:
        shutil.copytree(source_path, spec.dest_runs_path, dirs_exist_ok=True)


def fetch_dataset(spec: DatasetSpec, dry_run: bool) -> None:
    if not spec.source_root:
        raise ValueError(
            f"No source configured for {spec.name}. "
            f"Pass --{spec.name}-source or set the corresponding environment variable."
        )

    ensure_dest_parent(spec)
    if spec.source_root.startswith("gs://"):
        download_from_gcs(spec, dry_run=dry_run)
    else:
        copy_from_local(spec, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HELM and/or VHELM benchmark outputs for final_scripts."
    )
    parser.add_argument(
        "--dataset",
        choices=["helm", "vhelm", "all"],
        default="all",
        help="Which dataset to fetch.",
    )
    parser.add_argument(
        "--helm-version",
        default="v1.0.0",
        help="HELM suite version used by the minimal experiments.",
    )
    parser.add_argument(
        "--vhelm-version",
        default="v2.0.0",
        help="VHELM suite version used by the minimal experiments.",
    )
    parser.add_argument(
        "--helm-source",
        default=None,
        help=(
            "Source root for HELM benchmark_output. Can be a gs:// path or a local "
            "directory containing runs/<suite_version>."
        ),
    )
    parser.add_argument(
        "--vhelm-source",
        default=None,
        help=(
            "Source root for VHELM benchmark_output. Can be a gs:// path or a local "
            "directory containing runs/<suite_version>."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions without downloading/copying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_specs(args)

    selected = ["helm", "vhelm"] if args.dataset == "all" else [args.dataset]
    for dataset_name in selected:
        spec = specs[dataset_name]
        ensure_dataset_root(spec)
        print(f"\nPreparing {spec.name} ({spec.suite_version})")
        print(f"  source: {spec.source_root or '<not set>'}")
        print(f"  dest:   {spec.dest_runs_path}")
        fetch_dataset(spec, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
