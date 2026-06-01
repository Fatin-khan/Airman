from pathlib import Path

import clickhouse_connect
import pandas as pd


PROCESSED_FILE = Path("data/processed/air_quality_cleaned.csv")

CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "airman_user"
CLICKHOUSE_PASSWORD = "airman_password"
CLICKHOUSE_DATABASE = "airman"
CLICKHOUSE_TABLE = "air_quality"


def read_processed_data() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Processed file not found: {PROCESSED_FILE}. "
            "Run python -m src.pipeline first."
        )

    print(f"Reading processed data from {PROCESSED_FILE}...")
    df = pd.read_csv(PROCESSED_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    print(f"Processed data shape: {df.shape}")
    return df


def get_clickhouse_client():
    print("Connecting to ClickHouse...")

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
    )

    print("Connected to ClickHouse.")
    return client


def create_table(client) -> None:
    print("Creating ClickHouse table if it does not exist...")

    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DATABASE}.{CLICKHOUSE_TABLE}
        (
            city String,
            country String,
            date String,
            season String,
            pm2_5_concentration_ug_m_3 Float64,
            pm_station_3 Float64,
            dew_point_fahrenheit Float64,
            temperature_fahrenheit Float64,
            pm2_5 Float64,
            pm10_concentration_ug_m_3 Float64,
            so2_concentration_ug_m_3 Float64,
            no2_concentration_ug_m_3 Float64,
            co_concentration_ug_m_3 Float64,
            o3_concentration_ug_m_3 Float64,
            humidity Float64,
            pressure_hpa Float64,
            wind_direction String,
            wind_speed_m_s Float64,
            precipitation_mm_hourly Float64,
            precipitation_mm_cumulated Float64,
            pm_station_1 Float64,
            pm_us_post Float64,
            pm_station_2 Float64,
            dew_point_celsius Float64,
            temperature_celsius Float64,
            datetime DateTime
        )
        ENGINE = MergeTree
        ORDER BY datetime
        """
    )

    print("Table is ready.")


def load_data(client, df: pd.DataFrame) -> None:
    print("Clearing old data from ClickHouse table...")
    client.command(f"TRUNCATE TABLE {CLICKHOUSE_DATABASE}.{CLICKHOUSE_TABLE}")

    print("Loading DataFrame into ClickHouse...")
    client.insert_df(f"{CLICKHOUSE_DATABASE}.{CLICKHOUSE_TABLE}", df)

    row_count = client.query(
        f"SELECT count() FROM {CLICKHOUSE_DATABASE}.{CLICKHOUSE_TABLE}"
    ).result_rows[0][0]

    print(f"Data loaded successfully. Row count in ClickHouse: {row_count}")


def run_loader() -> None:
    df = read_processed_data()
    client = get_clickhouse_client()
    create_table(client)
    load_data(client, df)


if __name__ == "__main__":
    run_loader()