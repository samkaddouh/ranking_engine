# Candidate-Job Matching Engine

    This project is an end-to-end simulation of a marketplace ranking engine, similar to the systems used by Mercor, Upwork, or LinkedIn.
    It generates synthetic candidates and jobs, builds match features, trains an ML ranking model, and visualizes results through an interactive Streamlit dashboard.
    The goal is to demonstrate data science, ML modeling, experimentation, and product thinking for DS/ML roles involving ranking, recommendations, and labor-market intelligence.

# Features
## Synthetic Data Generation:

- 30,000+ candidates
- 5,000+ jobs
- 100,000+ candidate–job applications

## Realistic distributions for:

- Skills
- Seniority Levels
- Job Types
- Matching Difficulty

# Feature Engineering
## Extracted match features include:

- Skill overlap
- Experience compatibility
- Seniority alignment
- Job-family similarity

## Ranking Model

- XGBoost classifier predicting match quality
- Saved as a reusable model artifact (matching_xgb.pkl)

## Evaluation

- Precision@K
- Score distributions
- Best/worst ranked candidates
- Job-specific performance breakdowns

## Interactive Dashboard (Streamlit)

- Select a job
- Display Top-K candidates
- Inspect prediction scores
- Visualize ranking metrics

## How to Run
1️⃣ Create virtual environment
    ```python3 -m venv .venv```
    ```source .venv/bin/activate```
    ```pip install -r requirements.txt```

2️⃣ Generate synthetic data
    ```python src/data_simulation.py```

3️⃣ Build features
    ```python src/features.py```

4️⃣ Train the ranking model
    ```python src/model.py```

5️⃣ Launch the dashboard
    ```streamlit run app/streamlit_app.py```

# Precision@K Explained

- Precision@K answers one question:<br>

```“Among the top K items the model ranked highest, how many are actually good?”```<br>

In this project:
- “items” = candidates
- “good” = candidates where is_good_match = 1
- “top K” = the top K candidates ranked by the model’s predicted score<br>

Precision@K is simply:<br>
```(number of true good matches in the top K) / K```<br>


# Example:
Model returns top 10 candidates
In those 10, 3 are actually good matches (is_good_match = 1)
Then:
Precision@10 = 3 / 10 = 0.30 = 30%

# Formula:
```P@K = (# of correct matches in top K) / K```
<br>It focuses on the quality of the top of the ranked list, which is the most important part of marketplace ranking and recommendation systems.

## Why This Project Matters

This project showcases:

- Real-world ML pipeline design
- Ranking & recommendation thinking
- Feature engineering at scale
- Model training + evaluation
- Interactive, stakeholder-facing dashboards

