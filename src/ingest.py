from pathlib import Path
import pandas as pd


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """
    Read a CSV file with basic encoding fallback.
    """
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1")


def load_raw_air_quality_data(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load all CSV files from data/raw and combine them into one DataFrame.
    """
    csv_files = list(raw_data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    dataframes = []

    for file_path in csv_files:
        print(f"Reading: {file_path.name}")
        df = read_csv_safely(file_path)
        df["source_file"] = file_path.name
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)

    print(f"Loaded {len(combined_df):,} rows from {len(csv_files)} CSV file(s).")

    return combined_df