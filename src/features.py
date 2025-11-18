import pandas as pd
from pathlib import Path


def build_feature_table(
    candidates_path: str,
    jobs_path: str,
    applications_path: str
) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_path)
    jobs = pd.read_csv(jobs_path)
    apps = pd.read_csv(applications_path)

    df = (
        apps
        .merge(candidates, on="candidate_id", how="left")
        .merge(jobs, on="job_id", how="left", suffixes=("_cand", "_job"))
    )

    # One-hot seniority levels
    df = pd.get_dummies(df, columns=["seniority_level_cand", "seniority_level_job"], drop_first=True)

    # Example engineered features
    df["exp_over_job_difficulty"] = df["years_experience"] / (df["job_difficulty"] + 1)
    df["skill_overlap_scaled"] = df["skill_overlap"] / len(
        [c for c in df.columns if c.startswith("skill_")]
    )

    target = df["is_good_match"]

    # Drop leakage-y or ID columns
    drop_cols = [
        "application_id",
        "candidate_id",
        "job_id",
        "is_good_match",
        "true_match_score"
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = target

    return X, y, feature_cols


def main():
    root = Path(__file__).resolve().parents[1]
    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = build_feature_table(
        data_raw / "candidates.csv",
        data_raw / "jobs.csv",
        data_raw / "applications.csv",
    )

    X.to_parquet(data_processed / "X.parquet")
    y.to_csv(data_processed / "y.csv", index=False)
    pd.Series(feature_cols).to_csv(data_processed / "feature_cols.txt", index=False, header=False)

    print("Features saved to data/processed/")


if __name__ == "__main__":
    main()
