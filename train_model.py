"""Train a reproducible student-result prediction model."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    n_records = 300

    data = pd.DataFrame({
        "attendance": rng.integers(50, 101, size=n_records),
        "internal_marks": rng.integers(20, 101, size=n_records),
        "assignment_marks": rng.integers(30, 101, size=n_records),
        "previous_score": rng.integers(30, 101, size=n_records),
    })
    weighted_score = (
        0.25 * data["attendance"]
        + 0.35 * data["internal_marks"]
        + 0.20 * data["assignment_marks"]
        + 0.20 * data["previous_score"]
    )
    data["result"] = (weighted_score >= 60).astype(int)
    data.to_csv(ROOT / "student_results.csv", index=False)

    features = ["attendance", "internal_marks", "assignment_marks", "previous_score"]
    X, y = data[features], data["result"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    joblib.dump(model, ROOT / "student_result_model.pkl")

    metrics = {
        "accuracy": accuracy,
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "features": features,
        "random_seed": RANDOM_SEED,
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Model trained on {len(X_train)} records; tested on {len(X_test)} records.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
