This project is an end-to-end simulation of a marketplace ranking engine, similar to systems used by platforms like Mercor, Upwork, or LinkedIn.
It generates synthetic candidates and jobs, computes match features, trains a ranking model, and visualizes results through a Streamlit dashboard.
The goal is to showcase data science, experimentation, applied ML modeling, and product thinking—skill sets used in real DS/ML roles working on search, recommendations, and labor-market ranking systems.

🚀 Features:
✔ Synthetic Data Generation
    - 30,000+ candidates
    - 5,000+ jobs
    - 100,000+ candidate–job applications
Realistic distributions for:
    - skills
    - seniority levels
    - job families
    - matching difficulty
    
✔ Feature Engineering
    - Extracted match features include:
    - Skill overlap
    - Experience alignment
    - Seniority fit
    - Category (job-family) similarity

✔ Ranking Model
    - XGBoost classifier trained to predict match quality
    - Saved as a .pkl artifact for reuse
    
✔ Evaluation
    - Precision@K (ranking metric)
    - Score distributions
    - Top/Bottom ranked candidates
    - Job-specific performance breakdowns
    
✔ Interactive Streamlit Dashboard
    - Select any job
    - View Top-K recommended candidates
    - Inspect prediction scores
    - Visualize Precision@K and performance insights
