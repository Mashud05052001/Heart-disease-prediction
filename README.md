---
language: en
license: mit
library_name: scikit-learn
pipeline_tag: tabular-classification
tags:
  - scikit-learn
  - tabular-classification
  - machine-learning
  - healthcare
  - heart-disease
  - cardiovascular
  - binary-classification
  - random-forest
metrics:
  - accuracy
  - precision
  - recall
  - f1
model-index:
  - name: Heart Disease Prediction (RandomForestClassifier)
    results:
      - task:
          type: tabular-classification
          name: Tabular Classification
        dataset:
          name: heart-disease.csv (included in project repo)
          type: csv
        metrics:
          - name: Test Accuracy
            type: accuracy
            value: 0.87
          - name: 5-fold CV Accuracy (mean)
            type: accuracy
            value: 0.8479781421
          - name: 5-fold CV Precision (mean)
            type: precision
            value: 0.8215873016
          - name: 5-fold CV Recall (mean)
            type: recall
            value: 0.9272727273
          - name: 5-fold CV F1 (mean)
            type: f1
            value: 0.8705403543
---

# ❤️ Heart Disease Prediction (scikit-learn Random Forest)

A classic machine learning model that predicts the likelihood of **heart disease** from structured patient medical attributes (tabular data).  
This repository contains a **joblib-serialized scikit-learn model** trained and evaluated in an end-to-end Jupyter Notebook workflow.

## Model Details

- **Developed by:** brej-29
- **Model type:** `RandomForestClassifier` (scikit-learn)
- **Task:** Binary classification (tabular)
- **Output labels:**
  - `0` → No heart disease
  - `1` → Heart disease present
- **Saved artifact:** `heart_disease_model.joblib`
- **Training notebook:** `HeartDiseasePredictionProject.ipynb`
- **Source code / project repo:** https://github.com/brej-29/Logicmojo-AIML-Assignments-heart-disease-prediction-ml
- **License:** MIT

## Intended Use

### Direct Use
- Educational / portfolio demonstration of an end-to-end ML pipeline:
  - EDA → modeling → hyperparameter tuning → evaluation → persistence
- Research prototyping and experimentation with classical ML on healthcare-like tabular data

### Out-of-Scope Use (Important)
- **Not for clinical diagnosis**
- **Not a medical device**
- **Not validated for real-world patient care**
- Do not use this model as the sole basis for medical decisions.

## Training Data

The model was trained on a tabular dataset included in the project repository as `heart-disease.csv`.

- **Rows:** 303
- **Columns:** 14
  - **Features:** 13
  - **Target:** 1 (`target`)
- **Target distribution:**
  - `1`: 165
  - `0`: 138

### Features (Input Schema)

The model expects **13 columns**:

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (commonly encoded as 1 = male, 0 = female) |
| `cp` | Chest pain type (categorical encoded as integers) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar (binary) |
| `restecg` | Resting ECG results (categorical encoded as integers) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (binary) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of peak exercise ST segment (categorical encoded as integers) |
| `ca` | Number of major vessels (categorical encoded as integers) |
| `thal` | Thalassemia category (categorical encoded as integers) |

## Training Procedure

### Data Split
- `train_test_split(test_size=0.2)`
- Randomness controlled via `np.random.seed(42)` in the notebook

### Candidate Models Explored
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest

### Hyperparameter Tuning
- `RandomizedSearchCV` used to tune Random Forest
  - `cv=5`, `n_iter=20`
- Best Random Forest hyperparameters found in the notebook:
  - `n_estimators=210`
  - `max_depth=3`
  - `min_samples_split=4`
  - `min_samples_leaf=19`

### Final Model
The saved model (`heart_disease_model.joblib`) corresponds to:

- `RandomForestClassifier(n_estimators=210, max_depth=3, min_samples_split=4, min_samples_leaf=19)`

## Evaluation

### Baseline Test Accuracy (single 80/20 split)
- KNN: ~0.689
- Logistic Regression: ~0.885
- Random Forest: ~0.836

### Final Model Performance
- **Loaded saved model test accuracy:** **0.87**

### Cross-Validated Metrics (5-fold mean) — Final Random Forest
- **Accuracy:** 0.8479781421
- **Precision:** 0.8215873016
- **Recall:** 0.9272727273
- **F1:** 0.8705403543

Note: The notebook also visualizes confusion matrices and ROC curves for model comparison.

## How to Use

### 1) Install dependencies
- `pip install scikit-learn joblib pandas numpy huggingface_hub`

### Local Run and Improvement Scripts
```bash
pip install -r requirements.txt

python run_model.py --model heart_disease_model.joblib
python train_improved_model.py
python run_model.py --model heart_disease_model_improved.joblib
```

`train_improved_model.py` leaves the original model untouched and saves an improved local artifact as `heart_disease_model_improved.joblib`, plus a metrics report in `model_report.json`.

### Local GUI
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser. The GUI uses `heart_disease_model_improved.joblib` when it is available, and falls back to the original saved model otherwise.

### 2) Load the model from Hugging Face Hub
```python
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd

# Replace with your HF repo id, e.g. "brej-29/heart-disease-prediction-rf"
repo_id = "YOUR_USERNAME/YOUR_MODEL_REPO"

model_path = hf_hub_download(
    repo_id=repo_id,
    filename="heart_disease_model.joblib"
)

model = joblib.load(model_path)

# Example input (values are placeholders; use correctly-encoded values)
sample = pd.DataFrame([{
    "age": 57,
    "sex": 1,
    "cp": 0,
    "trestbps": 120,
    "chol": 354,
    "fbs": 0,
    "restecg": 1,
    "thalach": 163,
    "exang": 1,
    "oldpeak": 0.6,
    "slope": 2,
    "ca": 0,
    "thal": 2
}])

pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0, 1]  # probability of class "1"

print("Prediction:", int(pred))
print("P(heart disease):", float(proba))
```
## Input Requirements

- Provide all 13 feature columns  
- Ensure categorical features (`cp`, `restecg`, `slope`, `ca`, `thal`) follow the same integer encoding as used in training  
- Numeric types should be valid numbers (`int`/`float`); no missing values  

## Bias, Risks, and Limitations

- Small dataset (303 rows) → results may not generalize to broader populations  
- Encoding-dependent: categorical values must match training conventions  
- No clinical validation: metrics are from offline evaluation only  
- False negatives are possible (missed risk) — do not use for medical screening without rigorous validation  

## Environmental Impact

Training and evaluation were performed using classical ML methods on a small tabular dataset and are expected to have minimal compute and carbon impact (CPU-only, short runtime).

## Technical Specifications

- Framework: scikit-learn  
- Model format: joblib (`heart_disease_model.joblib`)  
- Inference type: CPU-friendly tabular prediction  

## Model Card Authors

- BrejBala

## Contact

For questions/feedback, please open an issue on the GitHub repository:  
https://github.com/brej-29/Logicmojo-AIML-Assignments-heart-disease-prediction-ml
# Heart-disease-prediction
