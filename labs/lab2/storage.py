"""Storage helpers for Lab 2.

Contains simple helpers to persist processed DataFrames to SQLite.
"""
from typing import Optional
import sqlite3
import pandas as pd


def save_to_sqlite(
    df: pd.DataFrame,
    db_path: str,
    table_name: str = "obesity_data_processed",
    if_exists: str = "replace",
    index: bool = False,
) -> None:
    """Save DataFrame to a SQLite database.

    Args:
        df: DataFrame to save.
        db_path: Path to sqlite database file.
        table_name: Table name to create/replace.
        if_exists: pandas.to_sql if_exists argument.
        index: Whether to write DataFrame index.
    """
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=index)
    finally:
        conn.close()
