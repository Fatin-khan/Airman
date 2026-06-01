from pathlib import Path
import pandas as pd
import re

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DATA_DIR / "air_quality_cleaned.csv"


def find_raw_csv() -> Path:
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    print(f"Found raw CSV file: {csv_files[0]}")
    return csv_files[0]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cleaned_columns = []

    for col in df.columns:
        col = col.strip().lower()
        col = col.replace("pm2.5", "pm2_5")
        col = re.sub(r"[^a-zA-Z0-9]+", "_", col)
        col = re.sub(r"_+", "_", col)
        col = col.strip("_")

        cleaned_columns.append(col)

    df.columns = cleaned_columns

    return df


def create_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_columns = {"year", "month", "day", "hour"}

    if required_columns.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]],
            errors="coerce"
        )

    elif "date" in df.columns:
        # First try flexible parsing. This is safer for your current dataset.
        df["datetime"] = pd.to_datetime(
             df["date"],
            errors="coerce",
            format="mixed"
        )

    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

    else:
        raise ValueError(
            "Could not create datetime column. Expected columns: "
            "year, month, day, hour OR date OR datetime"
        )

    failed_rows = df["datetime"].isna().sum()

    if failed_rows == len(df):
        print("Date parsing failed. Here are sample date values:")
        print(df["date"].head(10).tolist())
        raise ValueError("All datetime values failed to parse.")

    print(f"Datetime parsing failed rows: {failed_rows}")

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")

    return df 


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = df.select_dtypes(include=["number"]).columns
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns

    df[numeric_columns] = df[numeric_columns].interpolate(method="linear")
    df[numeric_columns] = df[numeric_columns].ffill().bfill()

    df[non_numeric_columns] = df[non_numeric_columns].ffill().bfill()

    return df


def remove_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columns_to_drop = ["no"]

    existing_columns_to_drop = [
        col for col in columns_to_drop if col in df.columns
    ]

    df = df.drop(columns=existing_columns_to_drop)

    return df


def run_pipeline() -> None:
    print("Starting Sprint 2 local ETL pipeline...")

    raw_file = find_raw_csv()

    print("Reading raw CSV...")
    df = pd.read_csv(raw_file)

    print(f"Raw data shape: {df.shape}")

    print("Cleaning column names...")
    df = clean_column_names(df)

    print("Creating datetime column...")
    df = create_datetime_column(df)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Removing useless columns...")
    df = remove_useless_columns(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving cleaned data to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)

    print("ETL pipeline completed successfully.")
    print(f"Cleaned data shape: {df.shape}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_pipeline()