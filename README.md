This project is an end-to-end simulation of a marketplace ranking engine, similar to the systems used by talent platforms like Mercor, Upwork, or LinkedIn. It generates synthetic candidates and jobs, extracts match features, trains a ranking model, and exposes the results through a Streamlit dashboard.

The objective is to demonstrate data science, experimentation, ML modeling, and product-focused reasoning—aligned with real DS/ML roles at companies working on labor markets, ranking systems, and LLM-powered matching engines.

🚀 Features:
    Synthetic data generator for:
    30,000+ candidates
    5,000+ jobs
    100k+ applications (candidate × job pairs)

Feature engineering pipeline to extract:
    Skill overlap
    Experience match
    Seniority alignment
    Category (job-family) similarity
    XGBoost ranking model predicting match quality
    Evaluation metrics:
    Precision@K
    Score distribution
    Worst and best matches
    Interactive Streamlit dashboard:
    Choose a job
    View top-K candidates
    Inspect model predictions
    Inspect Precision@K metrics

📁 Project Structure
ranking_engine/
│
├── app/
│   ├── streamlit_app.py         # Interactive dashboard
│
├── data/
│   ├── raw/                     # Generated raw datasets
│   ├── processed/               # Feature matrices, labels
│
├── artifacts/
│   ├── matching_xgb.pkl         # Trained XGBoost model
│
├── src/
│   ├── data_simulation.py       # Candidate/job/application generator
│   ├── features.py              # Feature engineering
│   ├── metrics.py               # Precision@K etc.
│   ├── model.py                 # Train model + save artifact
│   ├── evaluate.py              # Offline evaluation
│
├── requirements.txt
└── README.md

🛠️ How to Run the Project

1. Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Generate synthetic data
python src/data_simulation.py

3. Generate features
python src/features.py

4. Train the model
python src/model.py

5. Launch the dashboard
streamlit run app/streamlit_app.py

📊 Precision@K Explanation
Precision@K = proportion of true good matches found in the top-K ranked results:
P@K = \frac{\text{# of correct matches in top K}}{K}
This is the primary metric used in ranking, search, and recommendation systems.