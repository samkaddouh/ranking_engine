This project is an end-to-end simulation of a marketplace ranking engine, similar to the systems used by Mercor, Upwork, or LinkedIn.
It generates synthetic candidates and jobs, builds match features, trains an ML ranking model, and visualizes results through an interactive Streamlit dashboard.
The goal is to demonstrate data science, ML modeling, experimentation, and product thinking for DS/ML roles involving ranking, recommendations, and labor-market intelligence.
🚀 Features
🔧 Synthetic Data Generation
30,000+ candidates
5,000+ jobs
100,000+ candidate–job applications
Realistic distributions for:
skills
seniority levels
job families
matching difficulty
🧠 Feature Engineering
Extracted match features include:
Skill overlap
Experience compatibility
Seniority alignment
Job-family similarity
🤖 Ranking Model
XGBoost classifier predicting match quality
Saved as a reusable model artifact (matching_xgb.pkl)
📊 Evaluation
Precision@K
Score distributions
Best/worst ranked candidates
Job-specific performance breakdowns
🌐 Interactive Dashboard (Streamlit)
Select a job
Display Top-K candidates
Inspect prediction scores
Visualize ranking metrics
📁 Project Structure
ranking_engine/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   └── matching_xgb.pkl
│
├── src/
│   ├── data_simulation.py
│   ├── features.py
│   ├── metrics.py
│   ├── model.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
🛠️ How to Run
1️⃣ Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2️⃣ Generate synthetic data
python src/data_simulation.py
3️⃣ Build features
python src/features.py
4️⃣ Train the ranking model
python src/model.py
5️⃣ Launch the dashboard
streamlit run app/streamlit_app.py
📊 Precision@K Explained
Precision@K measures how many true good matches appear in the top-K ranked predictions.
Formula
P@K = (# of correct matches in top K) / K
It focuses on the quality of the top of the ranked list, which is the most important part of marketplace ranking and recommendation systems.
🎯 Why This Project Matters
This project showcases:
Real-world ML pipeline design
Ranking & recommendation thinking
Feature engineering at scale
Model training + evaluation
Interactive, stakeholder-facing dashboards
Applied DS/ML understanding
