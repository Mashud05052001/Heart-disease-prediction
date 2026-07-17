# Heart Disease Prediction Web App

A complete machine learning project for heart disease prediction using a tabular medical dataset, an improved scikit-learn model, and a polished Flask web interface.

The app predicts whether a patient profile is closer to the `0 = no heart disease` class or the `1 = heart disease present` class. It also shows the estimated probability of heart disease.

Important: this project is for education and demonstration only. It is not a medical device and must not be used for real clinical diagnosis.

## Live Website

| Page | Live URL | Description |
|---|---|---|
| Home | https://heart-disease-prediction-livid.vercel.app/ | Main landing page with project summary, model metrics, and navigation to prediction. |
| Project Overview | https://heart-disease-prediction-livid.vercel.app/project-overview | Web version of the lab report and project details, shown side by side with the live app description. |
| Prediction | https://heart-disease-prediction-livid.vercel.app/prediction | Interactive prediction interface for entering patient values and seeing the model result. |
| Health Check | https://heart-disease-prediction-livid.vercel.app/health | Simple endpoint to confirm the app and model are running. |

## Project Overview

This project includes:

- `heart-disease.csv` dataset with 303 rows and 14 columns, sourced from Hugging Face.
- Two saved model artifacts: `heart_disease_model.joblib` and `heart_disease_model_improved.joblib`.
- A Flask web app that loads the improved model by default.
- A prediction page that accepts 13 clinical features and returns a prediction plus probability.
- A dedicated `/project-overview` page that presents the lab report content in a readable web format.

## Project Features

- Heart disease prediction from 13 clinical input features.
- Improved model saved as `heart_disease_model_improved.joblib`.
- Original model preserved as `heart_disease_model.joblib`.
- Live prediction result updates automatically when input values change.
- Manual `Predict Risk` button remains available.
- `Low`, `Medium`, and `High` sample profiles are based on real dataset rows and produce different risk outputs.
- Light and dark theme support.
- Homepage with model summary and performance metrics.
- `/project-overview` page that renders the lab-report text file in a readable web format.
- Download option for the lab-report `.txt` file.

## Dataset

Dataset file:

`heart-disease.csv`

Dataset/model source:

[BrejBala/Heart-Disease-Prediction on Hugging Face](https://huggingface.co/BrejBala/Heart-Disease-Prediction)

Source note:

The local `heart-disease.csv` file used for training and evaluation comes from the Hugging Face repository/model card above. The source page documents the same tabular heart-disease data shape used in this project: `303` rows, `14` columns, `13` input features, and one `target` column. The Hugging Face repository is listed with the MIT license.

Dataset shape:

- Rows: `303`
- Columns: `14`
- Input features: `13`
- Target column: `target`

Target labels:

- `0` = No heart disease
- `1` = Heart disease present

Target distribution:

- `1`: `165`
- `0`: `138`

## Input Features

| Feature | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | Encoded sex value, commonly `1 = male`, `0 = female` |
| `cp` | Chest pain type, encoded as integers |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar, binary encoded |
| `restecg` | Resting ECG result, encoded as integers |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina, binary encoded |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels, encoded |
| `thal` | Thalassemia category, encoded |

The categorical values must use the same encoding as the training dataset.

## Model Information

### Original Model

File:

`heart_disease_model.joblib`

Model type:

`RandomForestClassifier`

Main hyperparameters:

- `n_estimators = 210`
- `max_depth = 3`
- `min_samples_split = 4`
- `min_samples_leaf = 19`

### Improved Model

File:

`heart_disease_model_improved.joblib`

Model type:

`Soft VotingClassifier(LogisticRegression pipeline + RandomForestClassifier)`

The improved model combines:

- A Logistic Regression pipeline with preprocessing.
- A tuned Random Forest classifier.

The Logistic Regression pipeline includes:

- Median imputation and scaling for numeric columns.
- Most-frequent imputation and one-hot encoding for categorical columns.

The improved model is the default model used by the web app.

## Model Performance

The models were evaluated using the same 80/20 train-test split:

- Training rows: `242`
- Test rows: `61`
- Random state: `42`

### Original Saved Model

| Metric | Value |
|---|---:|
| Accuracy | `0.8689` |
| Precision | `0.8529` |
| Recall | `0.9062` |
| F1 score | `0.8788` |
| ROC-AUC | `0.9364` |

Confusion matrix:

```text
[[24, 5],
 [ 3, 29]]
```

### Improved Model

| Metric | Value |
|---|---:|
| Accuracy | `0.9016` |
| Precision | `0.9062` |
| Recall | `0.9062` |
| F1 score | `0.9062` |
| ROC-AUC | `0.9375` |

Confusion matrix:

```text
[[26, 3],
 [ 3, 29]]
```

The improved model reduced false positives from `5` to `3` while keeping false negatives at `3` on the holdout test split.

### Improved Model 5-Fold Cross-Validation

| Metric | Mean Value |
|---|---:|
| Accuracy | `0.8647` |
| Precision | `0.8569` |
| Recall | `0.9152` |
| F1 score | `0.8828` |
| ROC-AUC | `0.9204` |

## Web App Pages

| Route | Description |
|---|---|
| `/` | Homepage with project introduction, hero visual, and model summary. |
| `/prediction` | Main prediction interface. Users can enter values and see instant model output. |
| `/project-overview` | Web formatted lab-report page generated from `Heart_Disease_Model_Lab_Report_Details.txt`. |
| `/project-overview/download` | Downloads the original lab-report text file. |
| `/predict` | JSON API endpoint used by the prediction page. |
| `/health` | Health check endpoint. |

The `/project-overview` page also highlights the Hugging Face dataset/model source used for `heart-disease.csv`.

## GUI Details

The frontend is built with HTML, CSS, and JavaScript using Flask templates.

Important UI files:

- `templates/index.html`
- `templates/prediction.html`
- `templates/project_overview.html`
- `static/style.css`
- `static/app.js`
- `static/home.js`
- `static/theme.js`

GUI features:

- Separate homepage and prediction page.
- Direct `/project-overview` page for lab-report explanation.
- Light/dark theme toggle.
- Instant prediction when any input value changes.
- Manual prediction button remains available.
- Reset button restores default sample values.
- ECG-style visual canvas on the prediction result panel.
- Probability gauge for heart disease likelihood.
- Model performance summary cards.

## Local Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask app locally:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Run Model Scripts

Evaluate the original model:

```bash
python run_model.py --model heart_disease_model.joblib
```

Train the improved model:

```bash
python train_improved_model.py
```

Evaluate the improved model:

```bash
python run_model.py --model heart_disease_model_improved.joblib
```

## Deployment Files

The project includes deployment helper files:

- `requirements.txt`
- `Procfile`
- `runtime.txt`
- `wsgi.py`
- `render.yaml`

Typical production start command:

```bash
gunicorn app:app
```
Files required at runtime:

- `heart_disease_model_improved.joblib`
- `heart_disease_model.joblib`
- `model_report.json`
- `Heart_Disease_Model_Lab_Report_Details.txt`
- `templates/`
- `static/`

## Important Limitations

- The dataset is small, with only `303` rows.
- The model is not clinically validated.
- The project is not a medical device.
- False positives and false negatives are possible.
- Encoded categorical values must match the dataset encoding.
- The result should be treated as a machine learning demonstration, not a medical diagnosis.

## Medical Disclaimer

This project is for learning, research practice, and portfolio demonstration only.

Do not use this app as the only basis for health decisions. If a person has chest pain, breathing difficulty, or any health concern, they should contact a qualified doctor or medical professional.

## Main Project Files

| File | Purpose |
|---|---|
| `app.py` | Main Flask backend and routes. |
| `heart-disease.csv` | Dataset used for training and evaluation. |
| `heart_disease_model.joblib` | Original saved Random Forest model. |
| `heart_disease_model_improved.joblib` | Improved saved model used by the app. |
| `train_improved_model.py` | Trains and saves the improved model. |
| `run_model.py` | Evaluates a saved model from the command line. |
| `model_report.json` | Stores model metrics and training details. |
| `Heart_Disease_Model_Lab_Report_Details.txt` | Detailed lab-report explanation. |
| `templates/` | HTML pages. |
| `static/` | CSS and JavaScript files. |

## License

MIT
