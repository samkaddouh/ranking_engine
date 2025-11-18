from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier


def train_model():
    root = Path(__file__).resolve().parents[1]
    data_processed = root / "data" / "processed"

    # Load processed features and target
    X = pd.read_parquet(data_processed / "X.parquet")
    y = pd.read_csv(data_processed / "y.csv")["is_good_match"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gradient boosting model (no XGBoost, fully scikit-learn)
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        max_iter=200,
        l2_regularization=0.0
    )

    model.fit(X_train, y_train)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    print(f"Validation AUC: {auc:.4f}")

    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    joblib.dump(model, artifacts_dir / "matching_xgb.pkl")  # keep same filename
    print("Model saved to artifacts/matching_xgb.pkl")


if __name__ == "__main__":
    train_model()
