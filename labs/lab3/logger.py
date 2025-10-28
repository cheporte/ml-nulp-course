"""Logging utilities for model predictions."""

import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Union


def log_predictions(
    model_name: str,
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    source: str = "train",
    db_path: str = '../../data/db/split_data.db'
) -> None:
    """Log model predictions to SQLite database.
    
    Args:
        model_name: Name of the model making predictions.
        y_true: True target values.
        y_pred: Predicted target values.
        source: Data source ('train' or 'test').
        db_path: Path to SQLite database for logging.
    """
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    actual_values = y_true.values if isinstance(y_true, pd.Series) else y_true

    log_df = pd.DataFrame({
        "timestamp": [datetime.now().isoformat()] * len(y_pred),
        "model": [model_name] * len(y_pred),
        "source": [source] * len(y_pred),
        "actual": actual_values,
        "predicted": y_pred
    })
    
    try:
        conn = sqlite3.connect(db_path)
        log_df.to_sql("predictions", conn, if_exists="append", index=False)
        conn.close()
        print(f"   -> Predictions for {model_name} ({source}) logged.")
    except sqlite3.Error as e:
        print(f"   -> Error logging predictions for {model_name} ({source}): {e}")