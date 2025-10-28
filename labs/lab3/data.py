"""Data loading and preprocessing utilities for Lab 3."""

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def load_and_split_data(
    test_size: float = 0.2,
    random_state: int = 42,
    db_path: str = '../../data/db/obesity_data_processed.db'
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    """Load data from SQLite and split into train/test sets.
    
    Args:
        test_size: Proportion of dataset to include in the test split.
        random_state: Random seed for reproducibility.
        db_path: Path to SQLite database.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) or (None, None, None, None) on error.
    """
    try:
        conn = sqlite3.connect(db_path)
        processed_data = pd.read_sql_query("SELECT * FROM obesity_data_processed", conn)
        conn.close()
    except sqlite3.Error as e:
        print(f"Error loading data from database: {e}")
        return None, None, None, None

    features = processed_data.drop(["NObeyesdad", "timestamp"], axis=1)
    target = processed_data["NObeyesdad"]
    
    # FIX: Reverse the MinMax scaling on the target variable
    target = (target * 6).round().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=test_size,
        random_state=random_state,
        stratify=target
    )
    print("Data loaded, target corrected, and split successfully.")
    return X_train, X_test, y_train, y_test