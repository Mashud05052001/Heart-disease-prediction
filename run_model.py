from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_model(path: Path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = joblib.load(path)

    version_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, InconsistentVersionWarning)
    ]
    return model, version_warnings


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | None]:
    predictions = model.predict(X_test)
    metrics: dict[str, float | None] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": None,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)

    return metrics


def print_metrics(metrics: dict[str, float | None]) -> None:
    for name, value in metrics.items():
        if value is None:
            continue
        print(f"{name:>10}: {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved heart disease model.")
    parser.add_argument("--model", default="heart_disease_model.joblib", help="Path to a joblib model artifact.")
    parser.add_argument("--data", default="heart-disease.csv", help="Path to the CSV dataset.")
    parser.add_argument("--sample-row", type=int, default=0, help="Dataset row to use for a sample prediction.")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {data_path}")

    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model, version_warnings = load_model(model_path)
    metrics = evaluate(model, X_test, y_test)
    predictions = model.predict(X_test)

    print(f"Model: {model_path}")
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target counts: {y.value_counts().to_dict()}")
    print(f"Notebook split: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")

    if version_warnings:
        print("\nCompatibility note:")
        print("This artifact was saved with a different scikit-learn version.")
        print("For local compatibility, retrain with: python train_improved_model.py")

    print("\nMetrics on test split:")
    print_metrics(metrics)

    print("\nConfusion matrix [[true_0, false_1], [false_0, true_1]]:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=4))

    sample = X.iloc[[args.sample_row]]
    sample_prediction = int(model.predict(sample)[0])
    print(f"Sample row {args.sample_row} input:")
    print(sample.to_string(index=False))
    print(f"Prediction: {sample_prediction} ({'heart disease present' if sample_prediction == 1 else 'no heart disease'})")

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(sample)[0, 1])
        print(f"P(heart disease): {probability:.4f}")


if __name__ == "__main__":
    main()
