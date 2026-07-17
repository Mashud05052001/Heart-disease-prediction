from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_improved_model() -> VotingClassifier:
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=210,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=19,
        random_state=RANDOM_STATE,
    )

    return VotingClassifier(
        estimators=[
            ("lr", logistic_pipeline),
            ("rf", random_forest),
        ],
        voting="soft",
    )


def build_reference_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=210,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=19,
        random_state=RANDOM_STATE,
    )


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities) if probabilities is not None else None,
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions, digits=4, output_dict=True),
    }
    return to_jsonable(metrics)


def cross_validated_metrics(model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, n_jobs: int) -> dict[str, float]:
    scores = cross_validate(clone(model), X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    return {
        metric: float(scores[f"test_{metric}"].mean())
        for metric in SCORING
    }


def load_original_model(path: Path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = joblib.load(path)

    version_warnings = sorted(
        {
            str(item.message).splitlines()[0]
            for item in caught
            if issubclass(item.category, InconsistentVersionWarning)
        }
    )
    return model, version_warnings


def print_metric_block(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n{title}")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        value = metrics.get(metric)
        if value is not None:
            print(f"{metric:>10}: {value:.4f}")
    print("confusion_matrix:")
    print(np.array(metrics["confusion_matrix"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an improved heart disease prediction model.")
    parser.add_argument("--data", default="heart-disease.csv", help="Path to the CSV dataset.")
    parser.add_argument("--original-model", default="heart_disease_model.joblib", help="Path to the original model.")
    parser.add_argument("--output-model", default="heart_disease_model_improved.joblib", help="Where to save the improved model.")
    parser.add_argument("--report", default="model_report.json", help="Where to save metrics and training details.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for cross-validation.")
    args = parser.parse_args()

    data_path = Path(args.data)
    original_model_path = Path(args.original_model)
    output_model_path = Path(args.output_model)
    report_path = Path(args.report)

    df = pd.read_csv(data_path)
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_improved_model(),
        param_grid={
            "lr__model__C": [0.23357214690901212, 1.0, 10.0],
            "lr__model__class_weight": [None, "balanced"],
            "weights": [(1, 1), (1, 2), (2, 1)],
        },
        scoring="f1",
        cv=cv,
        n_jobs=args.n_jobs,
        refit=True,
    )

    search.fit(X_train, y_train)
    improved_model = search.best_estimator_
    improved_metrics = evaluate(improved_model, X_test, y_test)
    improved_cv_metrics = cross_validated_metrics(improved_model, X, y, cv, args.n_jobs)

    original_metrics = None
    version_warnings = []
    if original_model_path.exists():
        original_model, version_warnings = load_original_model(original_model_path)
        original_metrics = evaluate(original_model, X_test, y_test)

    reference_rf_cv_metrics = cross_validated_metrics(build_reference_random_forest(), X, y, cv, args.n_jobs)

    joblib.dump(improved_model, output_model_path)

    report = {
        "dataset": str(data_path),
        "rows": len(df),
        "features": list(X.columns),
        "target_counts": y.value_counts().to_dict(),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
        "search": {
            "scoring": "f1",
            "best_params": search.best_params_,
            "best_cv_score_on_training_split": search.best_score_,
        },
        "original_model": {
            "path": str(original_model_path),
            "sklearn_version_warnings": version_warnings,
            "test_metrics": original_metrics,
            "reference_5_fold_cv_metrics": reference_rf_cv_metrics,
        },
        "improved_model": {
            "path": str(output_model_path),
            "model_type": "soft VotingClassifier(LogisticRegression pipeline + RandomForestClassifier)",
            "test_metrics": improved_metrics,
            "five_fold_cv_metrics": improved_cv_metrics,
        },
    }

    report_path.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")

    if original_metrics is not None:
        print_metric_block("Original saved model on notebook test split", original_metrics)
    print_metric_block("Improved model on notebook test split", improved_metrics)

    print("\nBest improved model params:")
    print(to_jsonable(search.best_params_))
    print(f"\nSaved improved model to: {output_model_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
