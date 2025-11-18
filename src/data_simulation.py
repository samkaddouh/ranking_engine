import numpy as np
import pandas as pd
from pathlib import Path


N_CANDIDATES = 30000
N_JOBS = 5000
RANDOM_SEED = 42

CANDIDATE_LEVELS = ["junior", "mid", "senior"]
JOB_LEVELS = ["junior", "mid", "senior"]
SKILLS = [
    "python", "sql", "statistics", "ml", "deep_learning",
    "nlp", "llm", "experimentation", "product_analytics"
]


def generate_candidates(n_candidates: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    candidate_ids = np.arange(1, n_candidates + 1)
    years_experience = rng.integers(0, 11, size=n_candidates)
    levels = pd.cut(
        years_experience,
        bins=[-1, 2, 6, 50],
        labels=CANDIDATE_LEVELS
    )

    base_skill_probs = np.array([0.8, 0.8, 0.7, 0.7, 0.4, 0.4, 0.3, 0.7, 0.7])
    skills_matrix = rng.binomial(1, base_skill_probs, size=(n_candidates, len(SKILLS)))

    candidates = pd.DataFrame({
        "candidate_id": candidate_ids,
        "years_experience": years_experience,
        "seniority_level": levels.astype(str),
    })

    for i, skill in enumerate(SKILLS):
        candidates[f"skill_{skill}"] = skills_matrix[:, i]

    return candidates


def generate_jobs(n_jobs: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1)

    job_ids = np.arange(1, n_jobs + 1)
    difficulty = rng.integers(1, 6, size=n_jobs)
    level_idx = rng.integers(0, len(JOB_LEVELS), size=n_jobs)
    levels = np.array(JOB_LEVELS)[level_idx]

    skills_matrix = rng.binomial(1, 0.5, size=(n_jobs, len(SKILLS)))

    jobs = pd.DataFrame({
        "job_id": job_ids,
        "job_difficulty": difficulty,
        "seniority_level": levels,
    })

    for i, skill in enumerate(SKILLS):
        jobs[f"required_{skill}"] = skills_matrix[:, i]

    return jobs


def simulate_applications(
    candidates: pd.DataFrame,
    jobs: pd.DataFrame,
    n_applications: int = 200000
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 2)

    candidate_ids = rng.choice(candidates["candidate_id"], size=n_applications)
    job_ids = rng.choice(jobs["job_id"], size=n_applications)

    apps = pd.DataFrame({
        "application_id": np.arange(1, n_applications + 1),
        "candidate_id": candidate_ids,
        "job_id": job_ids,
    })

    merged = (
        apps
        .merge(candidates, on="candidate_id", how="left")
        .merge(jobs, on="job_id", how="left", suffixes=("_cand", "_job"))
    )

    # Compute simple skill overlap
    skill_cols_cand = [c for c in merged.columns if c.startswith("skill_")]
    skill_cols_job = [c for c in merged.columns if c.startswith("required_")]

    skill_overlap = np.zeros(len(merged))
    for cand_col in skill_cols_cand:
        skill = cand_col.replace("skill_", "")
        job_col = f"required_{skill}"
        skill_overlap += merged[cand_col] * merged[job_col]

    merged["skill_overlap"] = skill_overlap

    # Seniority distance penalty
    level_map = {"junior": 0, "mid": 1, "senior": 2}
    merged["seniority_distance"] = (
        merged["seniority_level_cand"].map(level_map)
        - merged["seniority_level_job"].map(level_map)
    ).abs()

    # True match score (hidden ground truth)
    score = (
        0.3 * merged["skill_overlap"]
        - 0.2 * merged["seniority_distance"]
        - 0.05 * merged["job_difficulty"]
        + 0.02 * merged["years_experience"]
    )

    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    merged["true_match_score"] = score

    # Binary label: high quality match
    merged["is_good_match"] = (merged["true_match_score"] > 0.6).astype(int)

    return merged[[
        "application_id",
        "candidate_id",
        "job_id",
        "true_match_score",
        "is_good_match",
        "skill_overlap",
        "seniority_distance"
    ]]


def main():
    root = Path(__file__).resolve().parents[1]
    data_raw = root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(N_CANDIDATES)
    jobs = generate_jobs(N_JOBS)
    applications = simulate_applications(candidates, jobs)

    candidates.to_csv(data_raw / "candidates.csv", index=False)
    jobs.to_csv(data_raw / "jobs.csv", index=False)
    applications.to_csv(data_raw / "applications.csv", index=False)

    print("Synthetic data generated in data/raw/")


if __name__ == "__main__":
    main()
