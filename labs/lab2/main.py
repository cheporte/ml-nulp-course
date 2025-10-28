"""Refactored runner for Lab 2.

This script uses the small helper modules in the same package so the
logic is importable from other labs (e.g. lab3).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .data import load_data, fill_missing_values
from .preprocessing import encode_categorical_features, scale_numerical_features
from .storage import save_to_sqlite


def add_random_timestamp(df: pd.DataFrame, year: int = 2025) -> pd.DataFrame:
    """Add a `timestamp` column filled with random dates inside given year.

    Returns the same DataFrame with a new `timestamp` column.
    """
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime(f"{year}-12-31")

    n = len(df)
    # pandas.Timestamp.value is in ns; convert to seconds for randint
    low = int(start_date.value // 10**9)
    high = int(end_date.value // 10**9)
    random_dates = pd.to_datetime(np.random.randint(low, high + 1, n), unit="s")
    df["timestamp"] = random_dates
    return df


def run_lab2(
    csv_path: str = "./data/csv/ObesityDataSet_raw_and_data_sinthetic.csv",
    db_path: str = "./data/db/obesity_data_processed.db",
):
    csv_path = Path(csv_path)
    db_path = Path(db_path)

    data = load_data(str(csv_path))
    data = fill_missing_values(data)
    data = encode_categorical_features(data)
    data = scale_numerical_features(data)
    data = add_random_timestamp(data, year=2025)

    # Show head
    print(data.head())

    # Ensure parent dir exists for DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_sqlite(data, str(db_path), table_name="obesity_data_processed", if_exists="replace", index=False)


if __name__ == "__main__":
    run_lab2()
