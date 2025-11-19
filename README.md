📌 Candidate–Job Matching Engine (Mercor-Style Demo)
This project is an end-to-end simulation of a marketplace ranking engine, similar to systems used by platforms like Mercor, Upwork, or LinkedIn.
It generates synthetic candidates and jobs, computes match features, trains an ML ranking model, and exposes the results through an interactive Streamlit dashboard.
The goal is to demonstrate data science, ML modeling, experimentation, and product thinking for DS/ML roles focused on ranking, recommendations, and labor-market intelligence.
🚀 Features
🔧 Synthetic Data Generation
30,000+ candidates
5,000+ jobs
100,000+ applications (candidate × job pairs)
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
Inspect model predictions
Visualize ranking metrics
