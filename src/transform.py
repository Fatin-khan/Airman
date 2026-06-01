import re
import pandas as pd


def clean_column_name(column_name: str) -> str:
    """
    Convert column names into clean snake_case format.
    """
    column_name = column_name.strip().lower()

    column_name = column_name.replace("pm2.5", "pm25")
    column_name = column_name.replace("pm2_5", "pm25")
    column_name = column_name.replace("ug/m3", "ug_m_3")
    column_name = column_name.replace("°", "")

    column_name = re.sub(r"[^a-z0-9]+", "_", column_name)
    column_name = column_name.strip("_")

    return column_name


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize all column names.
    """
    df = df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    return df


def create_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a datetime column from available date/time columns.
    """
    df = df.copy()

    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(
    df["date"],
    format="mixed",
    errors="coerce"
)

    elif {"year", "month", "day", "hour"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            {
                "year": df["year"],
                "month": df["month"],
                "day": df["day"],
                "hour": df["hour"],
            },
            errors="coerce",
        )

    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    else:
        print("Warning: Could not create datetime column.")
        print("Available columns are:")
        print(df.columns.tolist())

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert measurement columns to numeric while keeping text columns safe.
    """
    df = df.copy()

    text_columns = {
        "city",
        "country",
        "date",
        "datetime",
        "season",
        "source_file",
        "station",
        "wind_direction",
        "wd",
        "cbwd",
    }

    for col in df.columns:
        if col not in text_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def create_pm25_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one unified PM2.5 target column for forecasting.
    """
    df = df.copy()

    if "pm25_concentration_ug_m_3" in df.columns and "pm25" in df.columns:
        df["target_pm25"] = df["pm25_concentration_ug_m_3"].combine_first(df["pm25"])

    elif "pm25_concentration_ug_m_3" in df.columns:
        df["target_pm25"] = df["pm25_concentration_ug_m_3"]

    elif "pm25" in df.columns:
        df["target_pm25"] = df["pm25"]

    else:
        print("Warning: No PM2.5 column found for target_pm25.")

    return df


def clean_air_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full transformation process.
    """
    df = standardize_columns(df)
    df = create_datetime_column(df)
    df = convert_numeric_columns(df)
    df = create_pm25_target_column(df)

    if "datetime" in df.columns:
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime")

    df = df.drop_duplicates()

    print(f"Cleaned data shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    if "datetime" in df.columns:
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    if "target_pm25" in df.columns:
        print(f"Rows with PM2.5 target: {df['target_pm25'].notna().sum():,}")

    return df