# ReconAI — Multi-Source Transaction Reconciler
Multi-Source Transaction Reconciler

# ReconAI — Multi-Source Transaction Reconciler

**AI Finance Controller for real-world cash reconciliation** — built for Razorpay Buildathon (Track 04).

## Problem

Finance teams manually reconcile ledger entries (internal books) against bank statements — a slow, error-prone process, especially when transactions don't match 1:1 (partial payments, batched settlements, consolidated transfers). ReconAI automates this using a trained classifier + an LLM reasoning agent, benchmarked on the real-world **BenchRec** cash reconciliation dataset (ICAIF 2023 Benchmark Competition).

## Approach

1. **Data pairing** — Reshaped labeled transactions (`matchId` groups) into A-side vs B-side pairs. Built confusing negative pairs via date/amount blocking to teach the model realistic distinctions.
2. **Feature engineering** — `amt_diff`, `amt_diff_pct_capped`, `date_diff`, `text_sim` (fuzzy text similarity).
3. **Classification model** — Logistic Regression trained to predict match probability. Confidence bucketed into **Auto-Match**, **Needs Review**, and **No Match**.
4. **Exception explanations** — Rule-based reasoning (`explain_pair`) surfaces *why* a pair needs review (e.g. amount mismatch, date gap).
5. **N:N reconciliation** — Real transactions often match many-to-many, not 1:1. Built tools (`find_candidates`, `check_consolidation`) to detect and verify N:N groups and consolidated payments.
6. **LLM reasoning agent** — A Google ADK agent (Gemini) uses the above tools to reason over ambiguous cases in natural language, going beyond fixed templates.
7. **Streamlit dashboard** — A self-contained app (`app.py`) exposing the same reconciliation logic for interactive use.

## Key Findings

- **Model performance:** 94% accuracy. Match: 91% precision / 99% recall. No Match: 99% precision / 86% recall.
- **Confidence bucketing reveals a threshold problem:** true matches cluster around 0.85–0.90 confidence, mostly *below* the 0.95 Auto-Match threshold — explaining why the Auto-Match bucket is small (1.1% of pairs) despite high underlying accuracy.
- **`date_diff` dominates the model** (coefficient ≈ -6.5) far more than amount-based features — this causes real edge cases, like an FX transaction with ~99.76% amount mismatch still being Auto-Matched purely on date alignment.
- **"Needs Review" isn't mostly model error** — 92.3% of it is genuine ambiguity from many-to-many (N:N) match structures the pairwise classifier can't fully disambiguate on its own (e.g. 19 candidate transactions, all true matches, all sharing an identical amount).
- **Consolidation-check limitation (honest finding):** cross-pairing every A with every B inside a `matchId` group can over-generate positive pairs when a group actually represents several separate 1:1 matches that happen to share the same amount — a known limitation of the labeling strategy, documented rather than hidden.
- **LLM reasoning agent validated:** on a known ambiguous case, the agent independently reproduced the human-verified finding *and* proposed a new disambiguation idea (checking transaction references) that wasn't hardcoded anywhere.

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/ReconAI-Rzp_Hackathon_2026.git
cd ReconAI-Rzp_Hackathon_2026
pip install -r requirements.txt
streamlit run app.py
```

The dashboard loads pre-trained artifacts from `artifacts/` (model, scaler, config, feature data) — no retraining needed to try it out.

To see the full pipeline (data prep → training → evaluation → agent reasoning), open `notebooks/ReconAI_Full_Pipeline.ipynb`.

> **Note:** The LLM reasoning agent requires a `GOOGLE_API_KEY` (Gemini) set as an environment variable. Without it, the dashboard's rule-based explanations still work fully — only the live agent reasoning (shown in the notebook) needs the key.

## Tech Stack

- **Data & ML:** pandas, numpy, scikit-learn, rapidfuzz
- **Agent framework:** Google ADK (Gemini `gemini-3.6-flash`)
- **Dashboard:** Streamlit
- **Visualization:** matplotlib, seaborn
- **Dataset:** [BenchRec cash reconciliation dataset](https://www.kaggle.com/) (ICAIF 2023 Benchmark Competition) — see `data/README.md`

## Repository Structure
