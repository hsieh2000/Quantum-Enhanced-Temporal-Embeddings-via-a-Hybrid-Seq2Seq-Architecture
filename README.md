# Quantum-Enhanced Temporal Embeddings via a Hybrid Seq2Seq Architecture (QCNC 2026)

## Project Overview

This repository contains code and data for training, inferring, and backtesting a quantum-inspired LSTM Seq2Seq model (QLSTM Seq2Seq) and an RBF-based backtesting strategy. It is organised to support quarter-by-quarter model training, latent-space extraction for use by the RBF strategy, and multi-period backtesting analysis.

## Contents

- `qlstm_seq2seq/` : QLSTM Seq2Seq model implementation, training and inference scripts.
- `rbf/` : RBF strategy code and backtesting utilities.
- `models/` : Trained model checkpoints and `model_logs.json` with metadata for each run.
- `data_collection/` : Raw data files and data collection scripts.
- `data/` : Preprocessed time-series used for training and evaluation (weekly aggregates present).
- `result/` : Latent CSV files and plots produced by the QLSTM inference step.
- `rbf/output/` : Backtesting outputs (equity curves, metrics, period folders).

## Requirements

Install Python dependencies listed in `requirements.txt` files. The top-level `requirements.txt` covers core packages; `qlstm_seq2seq/requirements.txt` may include additional project-specific packages.

Example:

```bash
python -m pip install -r requirements.txt
python -m pip install -r qlstm_seq2seq/requirements.txt
```

## Quick Usage

1) Train QLSTM Seq2Seq

Run quarterly training across the specified periods:

```bash
python qlstm_seq2seq/training.py
```

- This saves model checkpoints to the `models/` folder.
- The training meta-information for every saved checkpoint is appended to `models/model_logs.json`.

2) Run Inference (generate latent-space outputs and plots)

```bash
python qlstm_seq2seq/inference.py
```

- This generates latent CSV files and plots for the corresponding training period under `qlstm_seq2seq/result/`.
- Filenames for latent outputs start with `training_latent_data_mapping` and include the training period/date range.

3) Execute RBF backtesting

```bash
python rbf/rbf_strategy.py
```

- This consumes the latent CSVs (from `qlstm_seq2seq/result/`) and runs the RBF-based backtesting.
- Backtesting outputs are written to `rbf/output/`.

## Expected Outputs (details)

- Models: `models/` contains `.pth` checkpoint files. `models/model_logs.json` records hyperparameters, period, timestamp, and other metadata for reproducibility.
- Latent outputs: `qlstm_seq2seq/result/` contains CSV files and plots used by the RBF strategy. Filenames begin with `training_latent_data_mapping` followed by the period.
- RBF overall results: `rbf/output/combined_equity_curves.html`, `rbf/output/combined_equity_curves.csv`, and `rbf/output/summary_all_periods_metrics.csv` provide combined results across all backtest periods.
- Period-level results: Individual period folders are created under `rbf/output/` with per-period metrics, equity curves, and other diagnostics.

## Typical Workflow (recommended)

1. Prepare data: ensure `data/` and `data_collection/` files are up-to-date and preprocessed as expected by `qlstm_seq2seq/data_preprocessing.py`.
2. Train models quarter-by-quarter using `qlstm_seq2seq/training.py`.
3. For each trained model, run `qlstm_seq2seq/inference.py` to produce latent CSVs and diagnostic plots.
4. Feed the latent CSVs into `rbf/rbf_strategy.py` to run backtests and generate aggregated reports.

## Configuration & Parameters

- Training and inference parameters are set in `qlstm_seq2seq/training.py` and `qlstm_seq2seq/inference.py` respectively. Edit those files to change training windows, batch sizes, device configuration, or other hyperparameters.
- The RBF strategy parameters (kernel settings, rebalance frequency, transaction cost assumptions, etc.) are in `rbf/rbf_strategy.py` and helper modules inside `rbf/`.

## Tips & Troubleshooting

- GPU: If training with GPUs, verify the device selection logic in `qlstm_seq2seq/training.py` and that CUDA is available.
- Data alignment: Ensure the date ranges used during training match those used for inference and backtesting. Mismatched indices commonly cause errors during the RBF stage.
- Missing files: If a latent CSV is missing, re-run the corresponding inference period; log entries in `models/model_logs.json` can help locate which checkpoint corresponds to which period.

## Reproducibility

- `models/model_logs.json` is intended to store metadata for each saved model (timestamp, configuration, period). Use this file to map checkpoints to experimental conditions.

## Contact / Citation

If you use this code in research or production, please cite the project appropriately. For questions or issues, open an issue in the repository or contact the project owner.

---

File locations referenced in this README:

- `qlstm_seq2seq/training.py` — training entry point
- `qlstm_seq2seq/inference.py` — inference/latent extraction
- `rbf/rbf_strategy.py` — backtesting entry point
- `models/` — model checkpoints and `model_logs.json`
