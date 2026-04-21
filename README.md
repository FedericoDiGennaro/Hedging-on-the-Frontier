# Hedging on the Frontier: Learning New Tasks with Few Samples

Minimal scripts and bundled data to reproduce a small subset of the figures from the paper.

## What Is Included

- `modulus/plot_combined_modulus.py`
  - Recreates the combined upper/lower modulus plot.
- `transfer/plot_k_comparison.py`
  - Recreates the `helm_k_comparison` and `vhelm_k_comparison` plots.
- `transfer/prepare_transfer_data.py`
  - Optional helper to regenerate the bundled transfer summary `.npz` files if you have access to the full experiment caches from the original repo.
- `raw_data/modulus/...`
  - Bundled `.npz` files used by the modulus plot.
- `raw_data/transfer/...`
  - Bundled summary `.npz` files used by the transfer K-comparison plots.
- `raw_data/lm_arena_J26.csv`
  - Additional arena CSV retained for the arena part.
- `raw_data/download_data.py`
  - Optional helper to download raw HELM/VHELM benchmark runs.
- `plots/`
  - Output directory created by the plotting scripts. Generated figures are ignored by Git.

## Requirements

Install the minimal Python dependencies with:

```bash
cd final_scripts
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The required packages are:

- `numpy`
- `matplotlib`

Notes:

- `raw_data/download_data.py` uses only the standard library, but if you want to download from `gs://...` you also need the Google Cloud CLI (`gcloud`) installed.
- The bundled modulus and transfer plots do not require downloading raw benchmark outputs.
- The raw benchmark download folders under `raw_data/helm/` and `raw_data/vhelm/` are created on demand and ignored by Git.

## How To Run

From the `final_scripts` folder:

### 1. Modulus plot

```bash
python modulus/plot_combined_modulus.py
```

This writes:

- `plots/modulus/combined_upper_and_lower_modulus_linear_cropped.png`
- `plots/modulus/combined_upper_and_lower_modulus_linear_cropped.pdf`

### 2. Transfer K-comparison plots

```bash
python transfer/plot_k_comparison.py
```

This writes:

- `plots/transfer/helm_k_comparison.png`
- `plots/transfer/helm_k_comparison.pdf`
- `plots/transfer/vhelm_k_comparison.png`
- `plots/transfer/vhelm_k_comparison.pdf`

## Optional Data Download

If you want the raw HELM/VHELM benchmark runs locally, use:

```bash
python raw_data/download_data.py --dry-run
python raw_data/download_data.py --dataset helm
python raw_data/download_data.py --dataset vhelm --vhelm-source /path/to/benchmark_output
```

By default, HELM uses the public HELM Lite bucket. VHELM requires an explicit source path or the `VHELM_BENCHMARK_OUTPUT_PATH` environment variable.

Downloaded runs will be placed under:

- `raw_data/helm/benchmark_output/runs/v1.0.0/`
- `raw_data/vhelm/benchmark_output/runs/v2.0.0/`

## Optional Regeneration Of Bundled Summary Files

This script is only needed if you want to rebuild the bundled `.npz` transfer summaries instead of using the copies already committed here.

### Transfer summaries

```bash
python transfer/prepare_transfer_data.py
```

This expects access to the full adaptive-pruning cached results directory:

- `../experiments/adaptive_pruning/results/`

and rewrites the summary files under:

- `raw_data/transfer/helm/`
- `raw_data/transfer/vhelm/`

## Folder Layout

```text
final_scripts/
├── README.md
├── requirements.txt
├── modulus/
│   └── plot_combined_modulus.py
├── transfer/
│   ├── plot_k_comparison.py
│   └── prepare_transfer_data.py
├── raw_data/
│   ├── download_data.py
│   ├── lm_arena_J26.csv
│   ├── modulus/
│   └── transfer/
└── plots/
```
