from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path("data/processed/air_quality_cleaned.csv")

TARGET_COLUMN = "pm2_5"

FEATURE_COLUMNS = [
    "pm2_5",
    "pm10_concentration_ug_m_3",
    "so2_concentration_ug_m_3",
    "no2_concentration_ug_m_3",
    "co_concentration_ug_m_3",
    "o3_concentration_ug_m_3",
    "humidity",
    "pressure_hpa",
    "wind_speed_m_s",
    "precipitation_mm_hourly",
    "dew_point_celsius",
    "temperature_celsius",
]

SEQUENCE_LENGTH = 24


def load_hourly_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned data file not found: {DATA_PATH}\n"
            "Run Sprint 2 pipeline first using: python -m src.pipeline"
        )

    df = pd.read_csv(DATA_PATH)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    hourly_df = (
        df.groupby("datetime")[FEATURE_COLUMNS]
        .mean()
        .reset_index()
        .sort_values("datetime")
    )

    hourly_df = hourly_df.dropna()

    return hourly_df


def create_sliding_windows(data, sequence_length=SEQUENCE_LENGTH):
    x_values = []
    y_values = []

    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)

    for i in range(len(data) - sequence_length):
        x = data[i : i + sequence_length]
        y = data[i + sequence_length, target_index]

        x_values.append(x)
        y_values.append(y)

    return np.array(x_values), np.array(y_values)


def prepare_dataset():
    hourly_df = load_hourly_data()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hourly_df[FEATURE_COLUMNS])

    x, y = create_sliding_windows(scaled_data)

    return hourly_df, x, y, scaler


def main():
    hourly_df, x, y, scaler = prepare_dataset()

    print("Hourly time-series data prepared successfully.")
    print(f"Hourly data shape: {hourly_df.shape}")
    print(f"Date range: {hourly_df['datetime'].min()} to {hourly_df['datetime'].max()}")

    print("\nSelected feature columns:")
    print(FEATURE_COLUMNS)

    print("\nSliding-window dataset:")
    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")

    print("\nMeaning:")
    print(f"Each X sample uses previous {SEQUENCE_LENGTH} hours.")
    print(f"Each y value is the next-hour {TARGET_COLUMN} value.")


if __name__ == "__main__":
    main()