from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/air_quality_cleaned.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned data file not found: {DATA_PATH}\n"
            "Run Sprint 2 pipeline first using: python -m src.pipeline"
        )

    df = pd.read_csv(DATA_PATH)

    print("Cleaned data loaded successfully.")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    if "datetime" not in df.columns:
        print("\nWARNING: datetime column not found.")
        return

    df["datetime"] = pd.to_datetime(df["datetime"])

    print("\nDatetime range:")
    print("Start:", df["datetime"].min())
    print("End:", df["datetime"].max())

    print("\nTime-series checks:")
    total_rows = len(df)
    unique_datetimes = df["datetime"].nunique()
    duplicate_datetime_rows = total_rows - unique_datetimes

    print(f"Total rows: {total_rows}")
    print(f"Unique datetime values: {unique_datetimes}")
    print(f"Duplicate datetime rows: {duplicate_datetime_rows}")

    print("\nRows per datetime summary:")
    rows_per_datetime = df.groupby("datetime").size()
    print(rows_per_datetime.describe())

    print("\nTop 10 datetime values with most rows:")
    print(rows_per_datetime.sort_values(ascending=False).head(10))

    print("\nNumeric columns available for model:")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(numeric_cols)


if __name__ == "__main__":
    main()