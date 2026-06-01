from pathlib import Path

from src.ingest import load_raw_air_quality_data
from src.transform import clean_air_quality_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "air_quality_cleaned.csv"


def run_pipeline() -> None:
    """
    Run the full local ETL pipeline.
    """
    print("Starting Airman ETL pipeline...")

    raw_df = load_raw_air_quality_data(RAW_DATA_DIR)

    cleaned_df = clean_air_quality_data(raw_df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed file saved to: {OUTPUT_FILE}")
    print("ETL pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()