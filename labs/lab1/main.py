"""
Lab 1: Data Loading, Exploration, and Storage
This script loads a CSV dataset, performs basic data exploration,
and stores the data into a SQLite database.
"""

import pandas as pd
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def store_data_to_db(data: pd.DataFrame, db_path: str, table_name: str):
    """Store DataFrame to a SQLite database."""
    import sqlite3

    start_date = pd.to_datetime("2025-01-01")
    end_date = pd.to_datetime("2025-12-31")

    n = len(data)
    random_dates = pd.to_datetime(
        np.random.randint(start_date.value // 10**9, end_date.value // 10**9, n),
        unit="s",
    )
    data["timestamp"] = random_dates

    conn = sqlite3.connect(db_path)
    data.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


if __name__ == "__main__":
    # Load data
    data = load_data("./data/csv/ObesityDataSet_raw_and_data_sinthetic.csv")

    # Display first few rows
    print("Data Preview:")
    print(data.head())

    # Basic statistics
    print("\nBasic Statistics:")
    print(data.describe())

    # Store data to database
    store_data_to_db(data, "./data/sample_data.db", "sample_table")
    print("\nData stored to database successfully.")
