from pathlib import Path

import pandas as pd

from src.storage import get_storage


def inspect_csv(file_path: Path) -> None:
    print("=" * 80)
    print(f"File: {file_path.name}")

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1", low_memory=False)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print()

    print("Column names:")
    for column in df.columns:
        print(f"- {column}")

    print()
    print("Missing values per column:")
    missing_values = df.isnull().sum()

    missing_found = False
    for column, count in missing_values.items():
        if count > 0:
            print(f"- {column}: {count}")
            missing_found = True

    if not missing_found:
        print("No missing values found.")

    print()
    print("Preview:")
    print(df.head())


def main():
    storage = get_storage()
    csv_files = storage.list_raw_files("*.csv")

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/. "
            "Put your raw Beijing air quality CSV file inside data/raw/."
        )

    print(f"CSV files found: {len(csv_files)}")
    print()

    for csv_file in csv_files:
        inspect_csv(csv_file)

    print()
    print("Local raw data check completed successfully.")


if __name__ == "__main__":
    main()
