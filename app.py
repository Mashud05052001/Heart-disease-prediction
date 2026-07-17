from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "heart_disease_model_improved.joblib",
    BASE_DIR / "heart_disease_model.joblib",
]
REPORT_PATH = BASE_DIR / "model_report.json"
LAB_REPORT_PATH = BASE_DIR / "Heart_Disease_Model_Lab_Report_Details.txt"
SECTION_HEADING_PATTERN = re.compile(r"^(\d+)\.\s+(.+)$")

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
        "age": 67,
        "sex": 1,
        "cp": 2,
        "trestbps": 152,
        "chol": 212,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.8,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    },
    "lower": {
        "age": 58,
        "sex": 1,
        "cp": 0,
        "trestbps": 128,
        "chol": 259,
        "fbs": 0,
        "restecg": 0,
        "thalach": 130,
        "exang": 1,
        "oldpeak": 3.0,
        "slope": 1,
        "ca": 2,
        "thal": 3,
    },
    "elevated": {
        "age": 37,
        "sex": 0,
        "cp": 2,
        "trestbps": 120,
        "chol": 215,
        "fbs": 0,
        "restecg": 1,
        "thalach": 170,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "ca": 0,
        "thal": 2,
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

    @app.get("/project-overview")
    def project_overview_page():
        return render_template(
            "project_overview.html",
            lab_report=load_lab_report(),
            model_name=model_path.name,
            report=report,
        )

    @app.get("/project-overview/download")
    def download_project_overview():
        return send_file(
            LAB_REPORT_PATH,
            as_attachment=True,
            download_name=LAB_REPORT_PATH.name,
            mimetype="text/plain",
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


def load_lab_report() -> dict[str, Any]:
    if not LAB_REPORT_PATH.exists():
        return {
            "title": "Heart Disease Prediction Model",
            "subtitle": "Lab Report Details",
            "source": LAB_REPORT_PATH.name,
            "intro_blocks": [{"type": "paragraph", "text": "The lab report text file was not found."}],
            "sections": [],
            "section_count": 0,
            "word_count": 0,
        }

    text = LAB_REPORT_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    intro_lines: list[str] = []
    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None

    for line in lines:
        heading_match = SECTION_HEADING_PATTERN.match(line.strip())
        if heading_match and heading_match.group(2).isupper():
            if current_section is not None:
                current_section["blocks"] = make_report_blocks(current_section.pop("body_lines"))
                sections.append(current_section)
            current_section = {
                "number": heading_match.group(1),
                "title": heading_match.group(2).title(),
                "anchor": f"section-{heading_match.group(1)}",
                "body_lines": [],
            }
            continue

        if current_section is None:
            intro_lines.append(line)
        else:
            current_section["body_lines"].append(line)

    if current_section is not None:
        current_section["blocks"] = make_report_blocks(current_section.pop("body_lines"))
        sections.append(current_section)

    return {
        "title": "Heart Disease Prediction Model",
        "subtitle": "Lab Report Details",
        "source": LAB_REPORT_PATH.name,
        "intro_blocks": make_report_blocks(intro_lines),
        "sections": sections,
        "section_count": len(sections),
        "word_count": len(re.findall(r"\b\w+\b", text)),
    }


def make_report_blocks(lines: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    paragraph_lines: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = " ".join(line.strip() for line in paragraph_lines).strip()
        paragraph_lines.clear()
        if not text:
            return
        if text.endswith(":") and len(text) <= 90:
            blocks.append({"type": "label", "text": text})
        else:
            blocks.append({"type": "paragraph", "text": text})

    def flush_list() -> None:
        if not list_items:
            return
        blocks.append({"type": "list", "items": list_items.copy()})
        list_items.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            list_items.append(stripped[2:])
            continue

        flush_list()
        paragraph_lines.append(stripped)

    flush_paragraph()
    flush_list()
    return blocks


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
