"""Preprocessing helpers: encoding and scaling.

These functions operate on pandas DataFrames and return the modified
DataFrame. They intentionally keep the simple API from the original
lab2 script so they are easy to import in later labs.
"""
from typing import Dict
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using Label Encoding.

    Returns the same DataFrame after encoding. Also returns internal
    encoders via attribute `._label_encoders` on the DataFrame for
    possible later use (non-intrusive, optional).
    """
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # attach encoders for potential later reuse; not typical but handy
    try:
        setattr(df, "_label_encoders", label_encoders)
    except Exception:
        # If attribute cannot be set, ignore silently; behavior is optional
        pass

    return df


def scale_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical features using Min-Max Scaling.

    Modifies and returns DataFrame. Attaches scaler as attribute
    `._minmax_scaler` on the DataFrame for possible inspection.
    """
    scaler = MinMaxScaler()

    scaled_columns = df.select_dtypes(include=["int64", "float64"]).columns
    if len(scaled_columns) > 0:
        df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

    try:
        setattr(df, "_minmax_scaler", scaler)
    except Exception:
        pass

    return df
