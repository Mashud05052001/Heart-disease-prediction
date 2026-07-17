from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "heart_disease_model_improved.joblib",
    BASE_DIR / "heart_disease_model.joblib",
]
REPORT_PATH = BASE_DIR / "model_report.json"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

NUMERIC_RANGES = {
    "age": (18, 100),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0, 7),
}

CATEGORICAL_VALUES = {
    "sex": {0, 1},
    "cp": {0, 1, 2, 3},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3, 4},
    "thal": {0, 1, 2, 3},
}

SAMPLE_PROFILES = {
    "balanced": {
        "age": 55,
        "sex": 1,
        "cp": 1,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 153,
        "exang": 0,
        "oldpeak": 0.8,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    },
    "lower": {
        "age": 41,
        "sex": 0,
        "cp": 1,
        "trestbps": 105,
        "chol": 198,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    },
    "elevated": {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    },
}


def create_app() -> Flask:
    app = Flask(__name__)
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("No model artifact found. Train a model first.")

    model = joblib.load(model_path)
    report = load_report()

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            model_name=model_path.name,
            report=report,
        )

    @app.get("/prediction")
    def prediction_page():
        return render_template(
            "prediction.html",
            model_name=model_path.name,
            report=report,
            samples=SAMPLE_PROFILES,
        )

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "model": model_path.name})

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        try:
            features = parse_features(payload)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        frame = pd.DataFrame([features], columns=FEATURES)
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0, 1]) if hasattr(model, "predict_proba") else None
        risk = risk_band(probability, prediction)

        return jsonify(
            {
                "prediction": prediction,
                "prediction_label": "Heart disease present" if prediction == 1 else "No heart disease",
                "probability": probability,
                "risk": risk,
                "signals": summarize_inputs(features),
                "model": model_path.name,
            }
        )

    return app


def load_report() -> dict[str, Any]:
    if not REPORT_PATH.exists():
        return {}

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    model_report = report.get("improved_model") or report.get("original_model") or {}
    test_metrics = model_report.get("test_metrics", {})
    confusion_matrix = test_metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
    try:
        tn, fp = confusion_matrix[0]
        fn, tp = confusion_matrix[1]
    except (TypeError, ValueError, IndexError):
        tn = fp = fn = tp = 0

    return {
        "dataset": report.get("dataset", "heart-disease.csv"),
        "rows": report.get("rows", 0),
        "feature_count": len(report.get("features", [])),
        "target_counts": report.get("target_counts", {}),
        "split": report.get("split", {}),
        "search": report.get("search", {}),
        "model_path": model_report.get("path", ""),
        "model_type": model_report.get("model_type", "RandomForestClassifier"),
        "test_metrics": test_metrics,
        "cv_metrics": model_report.get("five_fold_cv_metrics")
        or model_report.get("reference_5_fold_cv_metrics", {}),
        "confusion": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def parse_features(payload: dict[str, Any]) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    for feature in FEATURES:
        if feature not in payload:
            raise ValueError(f"Missing value for {feature}")

        if feature in NUMERIC_RANGES:
            value = float(payload[feature])
            low, high = NUMERIC_RANGES[feature]
            if not low <= value <= high:
                raise ValueError(f"{feature} must be between {low} and {high}")
            parsed[feature] = round(value, 2) if feature == "oldpeak" else int(round(value))
            continue

        value = int(payload[feature])
        if value not in CATEGORICAL_VALUES[feature]:
            allowed = sorted(CATEGORICAL_VALUES[feature])
            raise ValueError(f"{feature} must be one of {allowed}")
        parsed[feature] = value

    return parsed


def risk_band(probability: float | None, prediction: int) -> dict[str, str]:
    if probability is None:
        return {
            "label": "Positive" if prediction == 1 else "Negative",
            "tone": "high" if prediction == 1 else "low",
            "summary": "Model classification only",
        }

    if probability >= 0.7:
        return {
            "label": "High estimated risk",
            "tone": "high",
            "summary": "Review with a qualified clinician",
        }
    if probability >= 0.4:
        return {
            "label": "Elevated estimated risk",
            "tone": "medium",
            "summary": "Several inputs resemble positive cases",
        }
    return {
        "label": "Lower estimated risk",
        "tone": "low",
        "summary": "Inputs lean toward lower model risk",
    }


def summarize_inputs(features: dict[str, float | int]) -> list[dict[str, str]]:
    return [
        {"label": "Age", "value": str(features["age"])},
        {"label": "Rest BP", "value": f"{features['trestbps']} mmHg"},
        {"label": "Cholesterol", "value": f"{features['chol']} mg/dL"},
        {"label": "Max HR", "value": str(features["thalach"])},
        {"label": "Exercise angina", "value": "Yes" if features["exang"] == 1 else "No"},
        {"label": "ST depression", "value": str(features["oldpeak"])},
    ]


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
