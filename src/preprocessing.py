"""
preprocessing.py
Helper functions for loading and preprocessing the diabetes dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os


TARGETS = ['diagnosed_diabetes', 'diabetes_stage', 'diabetes_risk_score']


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame):
    """
    Clean, encode, and scale the feature matrix.
    Returns X_scaled (numpy array), feature names, scaler, and label encoder.
    """
    X_raw = df.drop(columns=TARGETS)

    num_cols = X_raw.select_dtypes(include='number').columns.tolist()
    cat_cols = X_raw.select_dtypes(exclude='number').columns.tolist()

    # impute
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    X_num = pd.DataFrame(num_imputer.fit_transform(X_raw[num_cols]), columns=num_cols)
    X_cat = pd.DataFrame(cat_imputer.fit_transform(X_raw[cat_cols]), columns=cat_cols)

    # encode
    X_cat_enc = pd.get_dummies(X_cat, columns=cat_cols)
    X_processed = pd.concat([X_num, X_cat_enc], axis=1)

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    # encode multiclass target
    le = LabelEncoder()
    y_multi = le.fit_transform(df['diabetes_stage'])

    y_binary = df['diagnosed_diabetes'].values
    y_reg    = df['diabetes_risk_score'].values

    return X_scaled, X_processed.columns.tolist(), y_binary, y_multi, y_reg, scaler, le


def preprocess_single_row(row: dict, feature_names: list, scaler: StandardScaler) -> np.ndarray:
    """
    Preprocess a single input dict (from Streamlit) to match training features.
    row: dict of raw feature values
    feature_names: list of column names after one-hot encoding
    scaler: fitted StandardScaler
    """
    row_df = pd.DataFrame([row])

    num_cols = row_df.select_dtypes(include='number').columns.tolist()
    cat_cols = row_df.select_dtypes(exclude='number').columns.tolist()

    if cat_cols:
        row_df = pd.get_dummies(row_df, columns=cat_cols)

    # align columns with training data (fill missing with 0)
    row_aligned = row_df.reindex(columns=feature_names, fill_value=0)

    return scaler.transform(row_aligned)
