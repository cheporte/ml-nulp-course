import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the DataFrame."""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                # if data type is object, fill with mode
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                # if data type is numeric, fill with median
                df[col].fillna(df[col].median(), inplace=True)
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using Label Encoding."""
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df


def scale_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical features using Min-Max Scaling."""
    scaler = MinMaxScaler()

    scaled_columns = df.select_dtypes(include=["int64", "float64"]).columns
    df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

    return df


if __name__ == "__main__":
    # Load the dataset
    data = load_data("./data/csv/ObesityDataSet_raw_and_data_sinthetic.csv")

    # Fill missing values
    data = fill_missing_values(data)

    # Encode categorical features
    data = encode_categorical_features(data)

    # Scale numerical features
    data = scale_numerical_features(data)

    # Add a timestamp column with random dates in 2025
    start_date = pd.to_datetime("2025-01-01")
    end_date = pd.to_datetime("2025-12-31")

    n = len(data)
    random_dates = pd.to_datetime(
        np.random.randint(start_date.value // 10**9, end_date.value // 10**9, n),
        unit="s",
    )
    data["timestamp"] = random_dates

    # Display the first few rows of the processed DataFrame
    print(data.head())

    # Save the processed DataFrame to a database
    import sqlite3
    conn = sqlite3.connect("./data/db/obesity_data_processed.db")
    data.to_sql("obesity_data_processed", conn, if_exists="replace", index=False)
    conn.close()
