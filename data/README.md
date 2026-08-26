# Dataset

This project uses the **BenchRec cash reconciliation dataset** from the ICAIF 2023 Benchmark Competition (available on Kaggle).

Raw dataset files are **not included in this repository** due to size — download them directly from Kaggle if you want to reproduce the training pipeline.

## Files used

- `BenchRec_cash_v1.0_train.csv` (149,854 rows) — labeled ledger (A-side) vs bank statement (B-side) transactions, used for training.
- `BenchRec_cash_v1.0_eval.csv` (69,171 rows) — unlabeled evaluation set.
- `BenchRec_cash_v1.0_solution.csv` (32,048 rows) — hidden answer key for the eval set (not used in training).

## Structure

Each transaction belongs to a `matchId` group. A group can be:
- **1:1** — one A transaction matches one B transaction.
- **N:N** — multiple A and B transactions form a genuine many-to-many match (e.g. a consolidated payment, or several separate transactions sharing identical amounts).

See the main [README](../README.md) for how these files are used in the pipeline.
