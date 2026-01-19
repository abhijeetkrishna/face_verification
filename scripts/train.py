
# Train Logistic Regression for Face Verification
# Hyperparameter tuning done in notebook. Here we input the best parameter.
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# -------------------------------
# CONFIG
# -------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
C_VALUE = 10

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("model")


# -------------------------------
# DATA LOADING
# -------------------------------
def load_data(data_dir: Path):
    pairs = pd.read_csv(data_dir / "pairs.csv")
    embeddings = np.load(data_dir / "embeddings.npy")
    return pairs, embeddings


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
def build_features(pairs: pd.DataFrame, embeddings: np.ndarray):
    e1 = embeddings[pairs.idx1.values]
    e2 = embeddings[pairs.idx2.values]

    X = np.hstack([
        np.abs(e1 - e2),
        e1 * e2,
        np.sum(e1 * e2, axis=1, keepdims=True),
        np.linalg.norm(e1 - e2, axis=1, keepdims=True),
    ])

    y = pairs.same.values
    return X, y


# -------------------------------
# MODEL TRAINING
# -------------------------------
def train_model(X_train, y_train):
    model = LogisticRegression(
        C=C_VALUE,
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }


# -------------------------------
# SAVE ARTIFACTS
# -------------------------------
def save_artifacts(model, metrics: dict, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "logistic_regression.joblib")

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(model_dir / "config.json", "w") as f:
        json.dump({
            "model": "logistic_regression",
            "C": C_VALUE,
            "features": ["abs_diff", "prod", "cos_sim", "l2_dist"],
            "random_state": RANDOM_STATE
        }, f, indent=2)


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Loading data...")
    pairs, embeddings = load_data(DATA_DIR)

    print("Building features...")
    X, y = build_features(pairs, embeddings)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)

    print("Saving model...")
    save_artifacts(model, metrics, MODEL_DIR)

    print(f"Done. Model saved to: {MODEL_DIR.resolve()}")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    main()
