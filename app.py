import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="ReconAI — Reconciliation Assistant", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('artifacts/recon_model.pkl')
    scaler = joblib.load('artifacts/recon_scaler.pkl')
    with open('artifacts/recon_config.json') as f:
        config = json.load(f)
    pairs_features = pd.read_csv('artifacts/pairs_features_for_app.csv')
    return model, scaler, config, pairs_features

model, scaler, config, pairs_features = load_artifacts()
feature_cols = config['feature_cols']
AMT_CLOSE_THRESHOLD = config['AMT_CLOSE_THRESHOLD']
DATE_CLOSE_THRESHOLD = config['DATE_CLOSE_THRESHOLD']

st.title("💰 ReconAI — Multi-Source Transaction Reconciler")
st.caption("AI Finance Controller · Real-world N:N cash reconciliation, benchmarked on BenchRec")


def get_bucket(proba):
    if proba >= config['auto_match_threshold']:
        return "Auto-Match"
    elif proba >= config['needs_review_threshold']:
        return "Needs Review"
    else:
        return "No Match"


def find_candidates(b_id):
    candidates = pairs_features[
        (pairs_features['B_id'] == b_id) & (pairs_features['matchId'] != -1)
    ].copy()

    if candidates.empty:
        return None

    features = candidates[feature_cols].values
    features_scaled = scaler.transform(features)
    candidates['proba'] = model.predict_proba(features_scaled)[:, 1]
    candidates['bucket'] = candidates['proba'].apply(get_bucket)
    candidates = candidates.sort_values('proba', ascending=False)

    return candidates


def check_consolidation(b_id, candidates):
    true_matches = candidates[candidates['label'] == 1]

    if true_matches.empty:
        return None

    b_amount = true_matches['B_amount'].iloc[0]
    sum_a_amounts = true_matches['A_amount'].sum()
    diff = abs(b_amount - sum_a_amounts)
    diff_pct = diff / b_amount if b_amount != 0 else float('inf')
    is_consolidated = diff_pct <= 0.01

    return {
        "num_true_matches": len(true_matches),
        "b_amount": b_amount,
        "sum_of_a_amounts": sum_a_amounts,
        "difference_pct": diff_pct,
        "is_consolidated_payment": is_consolidated
    }


def explain_pair(row):
    amt_close = row['amt_diff_pct_capped'] <= AMT_CLOSE_THRESHOLD
    date_close = row['date_diff'] <= DATE_CLOSE_THRESHOLD

    if amt_close and date_close:
        return (f"Amount and date both align closely, but the model is still uncertain "
                f"(confidence {row['proba']:.0%}) — likely a genuinely ambiguous case, "
                f"possibly a duplicate candidate or near-tie with another transaction.")
    elif amt_close and not date_close:
        return (f"Amount matches closely ({row['amt_diff_pct_capped']:.2%} diff), but dates are "
                f"{row['date_diff']:.0f} days apart — check for a settlement delay.")
    elif date_close and not amt_close:
        return (f"Dates align closely, but amount differs by {row['amt_diff_pct_capped']:.2%} "
                f"— check for a partial payment, fees, or FX conversion.")
    else:
        return (f"Both amount ({row['amt_diff_pct_capped']:.2%} diff) and date "
                f"({row['date_diff']:.0f} days) show meaningful mismatch — flagged for "
                f"manual verification.")


st.divider()
b_id_input = st.text_input("Enter a B_id to check its reconciliation status:", "")

if st.button("Analyze") and b_id_input.strip():
    try:
        b_id = int(b_id_input.strip())
    except ValueError:
        st.error("Please enter a valid numeric B_id.")
        st.stop()

    candidates = find_candidates(b_id)

    if candidates is None:
        st.warning(f"No candidates found for B_id {b_id}.")
    else:
        top = candidates.iloc[0]
        bucket = top['bucket']
        proba = top['proba']

        bucket_color = {"Auto-Match": "green", "Needs Review": "orange", "No Match": "red"}
        st.markdown(f"### Status: :{bucket_color[bucket]}[{bucket}]  ·  Confidence: {proba:.1%}")

        st.subheader(f"Candidates found: {len(candidates)}")
        display_df = candidates[['A_id', 'proba', 'bucket', 'amt_diff_pct_capped', 'date_diff', 'label']].copy()
        display_df.columns = ['A_id', 'Confidence', 'Bucket', 'Amount Diff %', 'Date Diff (days)', 'Is True Match']
        st.dataframe(display_df, use_container_width=True)

        if len(candidates) > 1:
            st.subheader("Consolidation Check")
            consolidation = check_consolidation(b_id, candidates)
            if consolidation:
                if consolidation['is_consolidated_payment']:
                    st.success(f"✅ This looks like a consolidated payment — B_amount ({consolidation['b_amount']:.2f}) "
                               f"matches the sum of {consolidation['num_true_matches']} A-transactions "
                               f"({consolidation['sum_of_a_amounts']:.2f}).")
                else:
                    st.info(f"ℹ️ Not a consolidated payment (difference: {consolidation['difference_pct']:.1%}). "
                            f"If all candidates share the same amount, this may be a batch of separate "
                            f"1:1 matches rather than a true one-to-many consolidation — check reference "
                            f"numbers to disambiguate.")

        st.subheader("Explanation")
        top_row = top.copy()
        st.write(explain_pair(top_row))
