import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    root = Path(__file__).resolve().parents[1]
    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    artifacts = root / "artifacts"

    candidates = pd.read_csv(data_raw / "candidates.csv")
    jobs = pd.read_csv(data_raw / "jobs.csv")
    apps = pd.read_csv(data_raw / "applications.csv")
    X = pd.read_parquet(data_processed / "X.parquet")
    y = pd.read_csv(data_processed / "y.csv")["is_good_match"]
    feature_cols = pd.read_csv(data_processed / "feature_cols.txt", header=None)[0].tolist()
    model = joblib.load(artifacts / "matching_xgb.pkl")

    return candidates, jobs, apps, X, y, feature_cols, model


def main():
    st.title("Candidate–Job Matching Engine (Mercor-style Demo)")
    st.write(
        "This dashboard simulates a marketplace matching system. "
        "Scores are model predictions of match quality."
    )

    candidates, jobs, apps, X, y, feature_cols, model = load_data()

    # Sidebar controls
    job_id = st.sidebar.selectbox(
        "Select a Job",
        options=jobs["job_id"].tolist()
    )
    k = st.sidebar.slider("Top K candidates", min_value=5, max_value=50, value=10, step=5)

    # Filter applications for the chosen job
    job_apps = apps[apps["job_id"] == job_id]
    job_indices = job_apps.index

    X_job = X.iloc[job_indices]
    y_job = y.iloc[job_indices]

    scores = model.predict_proba(X_job[feature_cols])[:, 1]
    job_apps = job_apps.assign(predicted_score=scores, is_good_match=y_job.values)

    job_apps_sorted = job_apps.sort_values("predicted_score", ascending=False).head(k)

    st.subheader(f"Job {job_id} – Top {k} Candidates by Model Score")
    st.dataframe(job_apps_sorted[["candidate_id", "predicted_score", "is_good_match"]])

    # Compute a simple precision@K as a proxy metric
    precision_at_k = job_apps_sorted["is_good_match"].mean()
    st.metric(label=f"Precision@{k}", value=f"{precision_at_k:.2%}")

    # Show some job & candidate details
    with st.expander("View job details"):
        st.write(jobs[jobs["job_id"] == job_id])

    example_candidate_id = int(job_apps_sorted["candidate_id"].iloc[0])
    with st.expander("Top candidate profile"):
        st.write(candidates[candidates["candidate_id"] == example_candidate_id])


if __name__ == "__main__":
    main()
