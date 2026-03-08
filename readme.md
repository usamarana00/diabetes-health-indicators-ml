# Diabetes Health Indicators – ML Project

A complete machine learning pipeline for predicting diabetes outcomes using the Diabetes Health Indicators Dataset. Covers binary classification, multiclass classification, and regression — with an interactive Streamlit app for predictions.

---

## Project Structure

```
diabetes-health-indicators-ml/
├── data/
│   ├── raw/                   # Original CSV file
│   └── processed/             # Cleaned/scaled data (X_processed, y_binary, y_multi, y_reg)
├── notebooks/
│   └── cleaning.ipynb         # Full ML pipeline: EDA → Training → Tuning
├── models/                    # Serialized model artifacts (.joblib)
│   ├── binary_model.joblib
│   ├── multiclass_model.joblib
│   ├── regression_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── feature_names.joblib
├── src/
│   ├── preprocessing.py       # Data cleaning and encoding functions
│   └── evaluation.py          # Performance plotting functions
├── streamlit_app/
│   └── app.py                 # Streamlit prediction interface
├── reports/                   # Documentation & demo video
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**File:** `data/raw/diabetes_dataset.csv`
**Rows:** 100,000 | **Columns:** 31

Includes clinical, lifestyle, and diagnostic data:
- Demographics: age, gender, ethnicity, education, income, employment
- Lifestyle: physical activity, diet score, sleep, screen time, alcohol, smoking
- Clinical: BMI, blood pressure, cholesterol, glucose, insulin, HbA1c

**Target columns:**
| Task | Target Column | Type |
|------|--------------|------|
| Binary Classification | `diagnosed_diabetes` | 0 / 1 |
| Multiclass Classification | `diabetes_stage` | 5 classes |
| Regression | `diabetes_risk_score` | continuous (2.7 – 67.2) |

---

## ML Pipeline (notebook: `cleaning.ipynb`)

### Phase 1 – Imports
All libraries loaded: pandas, numpy, sklearn, matplotlib, seaborn, joblib.

### Phase 2 – Load & Inspect
- Shape: (100,000, 31)
- No missing values
- Mixed dtypes: 16 int64, 8 float64, 7 categorical (str)

### Phase 3 – Preprocessing
- Imputed with mean (numeric) / most_frequent (categorical)
- One-hot encoded 6 categorical columns → 46 total features
- Standardized with `StandardScaler`
- Multiclass target label-encoded with `LabelEncoder`

### Phase 4 – EDA
- Target distribution plots (bar charts, histogram)
- Feature histograms (age, BMI, HbA1c, glucose, risk score)
- Correlation heatmap
- Boxplots: BMI / HbA1c / fasting glucose vs diagnosis
- Group-wise mean risk score by diabetes stage

### Phase 5 – Data Splitting
- 80/20 train/test split, stratified for classification tasks

### Phase 6 – Baseline Models & Results

**Binary Classification (`diagnosed_diabetes`)**
| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Logistic Regression | 0.8607 | 0.8852 | 0.9337 |
| Decision Tree | 0.8581 | 0.8830 | 0.8495 |
| KNN | 0.7630 | 0.8045 | 0.8274 |

**Multiclass Classification (`diabetes_stage`)**
| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Decision Tree | 0.8549 | 0.5145 |
| Logistic Regression | 0.8194 | 0.4665 |
| KNN | 0.6990 | 0.3629 |

**Regression (`diabetes_risk_score`)**
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.4391 | 0.7445 | 0.9933 |
| Decision Tree Regressor | 1.2467 | 1.5905 | 0.9693 |

### Phase 7 – Hyperparameter Tuning (GridSearchCV, 5-fold CV)

Tuned `DecisionTreeClassifier` / `DecisionTreeRegressor` with param grid:
- `max_depth`: [3, 5, 10, None]
- `min_samples_split`: [2, 5, 10]

| Task | Metric | Baseline | Tuned | Best Params |
|------|--------|----------|-------|-------------|
| Binary | F1 | 0.8830 | **0.9284** | depth=5, split=2 |
| Multiclass | Macro-F1 | 0.5145 | **0.5506** | depth=5, split=10 |
| Regression | R² | 0.9693 | **0.9719** | depth=None, split=10 |

### Phase 8 – Model Saving
All tuned models and preprocessing artifacts saved to `models/` with `joblib.dump()`.

---

## Streamlit App

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run streamlit_app/app.py
```

App opens at: `http://localhost:8501`

### Features
- **Sidebar** — input form for all 28 patient features (sliders + dropdowns)
- **Tab 1: Binary Classification** — predicts Diabetic / Not Diabetic with confidence score
- **Tab 2: Multiclass Classification** — predicts diabetes stage with class probability table
- **Tab 3: Regression** — predicts continuous risk score with Low / Moderate / High risk label

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Key Results Summary

- **Best binary classifier:** Tuned Decision Tree — F1 = **0.9284**, ROC-AUC = 0.9337 (LR baseline)
- **Best multiclass classifier:** Tuned Decision Tree — Macro-F1 = **0.5506**, Accuracy = 0.8549
- **Best regression model:** Linear Regression — R² = **0.9933**, RMSE = 0.7445
- Models show strong performance on this 100K-row healthcare dataset
