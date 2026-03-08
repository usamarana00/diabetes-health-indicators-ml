"""
app.py
Streamlit app for the Diabetes Health Indicators ML Project.
Three tabs: Binary Classification, Multiclass Classification, Regression.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── page config MUST be first Streamlit call ──────────────────────────────────
st.set_page_config(page_title="Diabetes ML Predictor", layout="wide")

# ── load models ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_artifacts():
    binary_model    = joblib.load(os.path.join(MODEL_DIR, 'binary_model.joblib'))
    multi_model     = joblib.load(os.path.join(MODEL_DIR, 'multiclass_model.joblib'))
    reg_model       = joblib.load(os.path.join(MODEL_DIR, 'regression_model.joblib'))
    scaler          = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    le              = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    feature_names   = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
    return binary_model, multi_model, reg_model, scaler, le, feature_names

binary_model, multi_model, reg_model, scaler, le, feature_names = load_artifacts()


# ── helpers ───────────────────────────────────────────────────────────────────
def build_input_vector(inputs: dict) -> np.ndarray:
    """One-hot encode categorical inputs and align to training feature columns."""
    row_df = pd.DataFrame([inputs])
    num_cols = row_df.select_dtypes(include='number').columns.tolist()
    cat_cols = row_df.select_dtypes(exclude='number').columns.tolist()

    if cat_cols:
        row_df = pd.get_dummies(row_df, columns=cat_cols)

    # fill missing columns with 0 (handles unseen categories)
    row_aligned = row_df.reindex(columns=feature_names, fill_value=0)
    return scaler.transform(row_aligned)


def input_form():
    """Sidebar form with all feature inputs. Returns a dict."""
    st.sidebar.header("Patient Information")

    age        = st.sidebar.slider("Age", 18, 90, 45)
    bmi        = st.sidebar.slider("BMI", 15.0, 40.0, 25.0, step=0.1)
    hba1c      = st.sidebar.slider("HbA1c (%)", 4.0, 10.0, 6.5, step=0.1)
    glucose_f  = st.sidebar.slider("Fasting Glucose (mg/dL)", 60, 180, 110)
    glucose_pp = st.sidebar.slider("Postprandial Glucose (mg/dL)", 70, 300, 160)
    insulin    = st.sidebar.slider("Insulin Level (μU/mL)", 2.0, 35.0, 9.0, step=0.1)
    systolic   = st.sidebar.slider("Systolic BP (mmHg)", 80, 200, 120)
    diastolic  = st.sidebar.slider("Diastolic BP (mmHg)", 50, 130, 80)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 120, 72)
    chol_total = st.sidebar.slider("Total Cholesterol (mg/dL)", 100, 320, 185)
    hdl        = st.sidebar.slider("HDL Cholesterol (mg/dL)", 20, 100, 54)
    ldl        = st.sidebar.slider("LDL Cholesterol (mg/dL)", 50, 270, 103)
    trigly     = st.sidebar.slider("Triglycerides (mg/dL)", 30, 350, 121)
    whr        = st.sidebar.slider("Waist-to-Hip Ratio", 0.70, 1.00, 0.85, step=0.01)
    activity   = st.sidebar.slider("Physical Activity (min/week)", 0, 840, 120)
    diet       = st.sidebar.slider("Diet Score (0–10)", 0.0, 10.0, 6.0, step=0.1)
    sleep      = st.sidebar.slider("Sleep Hours/Day", 3.0, 10.0, 7.0, step=0.1)
    screen     = st.sidebar.slider("Screen Time (hrs/day)", 0.5, 17.0, 6.0, step=0.5)
    alcohol    = st.sidebar.slider("Alcohol (drinks/week)", 0, 10, 2)

    fam_hist   = st.sidebar.selectbox("Family History of Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
    hypert     = st.sidebar.selectbox("Hypertension History", [0, 1], format_func=lambda x: "Yes" if x else "No")
    cardio     = st.sidebar.selectbox("Cardiovascular History", [0, 1], format_func=lambda x: "Yes" if x else "No")

    gender     = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    ethnicity  = st.sidebar.selectbox("Ethnicity", ["Asian", "Black", "Hispanic", "Other", "White"])
    education  = st.sidebar.selectbox("Education Level", ["Graduate", "Highschool", "No formal", "Postgraduate"])
    income     = st.sidebar.selectbox("Income Level", ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"])
    employment = st.sidebar.selectbox("Employment Status", ["Employed", "Retired", "Student", "Unemployed"])
    smoking    = st.sidebar.selectbox("Smoking Status", ["Current", "Former", "Never"])

    return {
        "age": age, "bmi": bmi, "hba1c": hba1c,
        "glucose_fasting": glucose_f, "glucose_postprandial": glucose_pp,
        "insulin_level": insulin, "systolic_bp": systolic, "diastolic_bp": diastolic,
        "heart_rate": heart_rate, "cholesterol_total": chol_total,
        "hdl_cholesterol": hdl, "ldl_cholesterol": ldl, "triglycerides": trigly,
        "waist_to_hip_ratio": whr,
        "physical_activity_minutes_per_week": activity,
        "diet_score": diet, "sleep_hours_per_day": sleep,
        "screen_time_hours_per_day": screen,
        "alcohol_consumption_per_week": alcohol,
        "family_history_diabetes": fam_hist,
        "hypertension_history": hypert,
        "cardiovascular_history": cardio,
        "gender": gender, "ethnicity": ethnicity,
        "education_level": education, "income_level": income,
        "employment_status": employment, "smoking_status": smoking
    }


# ── page setup ────────────────────────────────────────────────────────────────
st.title("Diabetes Health Indicators – ML Predictor")
st.write("Use the sidebar to enter patient information, then select a prediction task below.")

inputs = input_form()
X_input = build_input_vector(inputs)

tab1, tab2, tab3 = st.tabs(["Binary Classification", "Multiclass Classification", "Regression"])


# ── Tab 1: Binary ─────────────────────────────────────────────────────────────
with tab1:
    st.header("Binary Classification: Diabetes Diagnosis")
    st.write("Predicts whether the patient has been diagnosed with diabetes (Yes/No).")

    if st.button("Predict (Binary)", key="btn_binary"):
        pred = binary_model.predict(X_input)[0]
        prob = binary_model.predict_proba(X_input)[0]

        result = "Diabetic" if pred == 1 else "Not Diabetic"
        confidence = prob[pred]

        if pred == 1:
            st.error(f"Prediction: **{result}**")
        else:
            st.success(f"Prediction: **{result}**")

        st.metric("Confidence", f"{confidence:.1%}")
        st.write(f"Probability (No Diabetes): {prob[0]:.1%} | Probability (Diabetes): {prob[1]:.1%}")


# ── Tab 2: Multiclass ─────────────────────────────────────────────────────────
with tab2:
    st.header("Multiclass Classification: Diabetes Stage")
    st.write("Predicts the stage of diabetes: No Diabetes, Pre-Diabetes, or Type 2.")

    if st.button("Predict (Stage)", key="btn_multi"):
        pred = multi_model.predict(X_input)[0]
        probs = multi_model.predict_proba(X_input)[0]

        stage = le.inverse_transform([pred])[0]
        confidence = probs[pred]

        if stage == "No Diabetes":
            st.success(f"Predicted Stage: **{stage}**")
        elif stage == "Pre-Diabetes":
            st.warning(f"Predicted Stage: **{stage}**")
        else:
            st.error(f"Predicted Stage: **{stage}**")

        st.metric("Confidence", f"{confidence:.1%}")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Stage": le.classes_,
            "Probability": [f"{p:.1%}" for p in probs]
        })
        st.dataframe(prob_df, use_container_width=True)


# ── Tab 3: Regression ─────────────────────────────────────────────────────────
with tab3:
    st.header("Regression: Diabetes Risk Score")
    st.write("Predicts the continuous diabetes risk score (range: ~2.7 – 67.2).")

    if st.button("Predict (Risk Score)", key="btn_reg"):
        pred_score = reg_model.predict(X_input)[0]

        st.metric("Predicted Risk Score", f"{pred_score:.2f}")

        # simple risk category
        if pred_score < 20:
            st.success("Risk Level: Low")
        elif pred_score < 35:
            st.warning("Risk Level: Moderate")
        else:
            st.error("Risk Level: High")

        st.caption("Risk Score scale: 0 = lowest risk, ~67 = highest risk based on training data.")
