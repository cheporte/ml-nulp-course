"""Data loading and simple cleaning utilities for Lab 2.

Functions:
- load_data(file_path) -> pd.DataFrame
- fill_missing_values(df) -> pd.DataFrame
"""
from typing import Any
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file.

    Args:
        file_path: Path to CSV file.

    Returns:
        DataFrame loaded from CSV.
    """
    return pd.read_csv(file_path)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the DataFrame.

    - For object dtypes, fills with mode.
    - For numeric dtypes, fills with median.
    The function mutates the DataFrame in place and returns it for
    convenience.
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                # if data type is object, fill with mode
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                # if data type is numeric, fill with median
                df[col].fillna(df[col].median(), inplace=True)
    return df
