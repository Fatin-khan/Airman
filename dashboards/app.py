from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Airman Dashboard",
    page_icon="🌫️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "air_quality_cleaned.csv"
PREDICTIONS_PATH = BASE_DIR / "reports" / "validation_predictions.csv"
METRICS_PATH = BASE_DIR / "reports" / "training_metrics.json"


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_air_quality_data():
    if not PROCESSED_DATA_PATH.exists():
        return None

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Try to detect datetime column
    datetime_candidates = ["datetime", "date", "timestamp"]
    datetime_col = None

    for col in datetime_candidates:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col is not None:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        df = df.dropna(subset=[datetime_col])
        df = df.sort_values(datetime_col)

    return df


@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return None

    pred_df = pd.read_csv(PREDICTIONS_PATH)

    if "datetime" in pred_df.columns:
        pred_df["datetime"] = pd.to_datetime(pred_df["datetime"], errors="coerce")
        pred_df = pred_df.dropna(subset=["datetime"])
        pred_df = pred_df.sort_values("datetime")

    return pred_df


@st.cache_data
def load_model_metrics():
    if not METRICS_PATH.exists():
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


# -----------------------------
# Header
# -----------------------------
st.title("🌫️ Airman — Air Quality Forecasting Dashboard")

st.write(
    "This dashboard shows historical air quality trends and PM2.5 forecasting "
    "results using the Airman ETL pipeline, ClickHouse-ready processed data, "
    "and a PyTorch LSTM model."
)

st.divider()


# -----------------------------
# Load data
# -----------------------------
df = load_air_quality_data()

if df is None:
    st.error(
        "Processed data file not found. Please run the ETL pipeline first.\n\n"
        "Expected file: data/processed/air_quality_cleaned.csv"
    )
    st.stop()


datetime_col = find_column(df, ["datetime", "date", "timestamp"])
pm25_col = find_column(
    df,
    [
        "pm2_5",
        "pm2_5_concentration_ug_m_3",
        "pm25",
        "PM2.5"
    ]
)

if datetime_col is None:
    st.error("No datetime/date column found in the processed dataset.")
    st.stop()

if pm25_col is None:
    st.error("No PM2.5 column found in the processed dataset.")
    st.stop()


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Dashboard Filters")

min_date = df[datetime_col].min().date()
max_date = df[datetime_col].max().date()

selected_date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered_df = df.copy()

if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_df = filtered_df[
        (filtered_df[datetime_col].dt.date >= start_date)
        & (filtered_df[datetime_col].dt.date <= end_date)
    ]


city_col = find_column(df, ["city", "City"])

if city_col is not None:
    city_options = sorted(df[city_col].dropna().unique().tolist())
    selected_cities = st.sidebar.multiselect(
        "Select city",
        options=city_options,
        default=city_options
    )

    if selected_cities:
        filtered_df = filtered_df[filtered_df[city_col].isin(selected_cities)]


# -----------------------------
# Summary metrics
# -----------------------------
st.subheader("Overview")

avg_pm25 = filtered_df[pm25_col].mean()
max_pm25 = filtered_df[pm25_col].max()
min_pm25 = filtered_df[pm25_col].min()
total_records = len(filtered_df)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average PM2.5", f"{avg_pm25:.2f}")
col2.metric("Maximum PM2.5", f"{max_pm25:.2f}")
col3.metric("Minimum PM2.5", f"{min_pm25:.2f}")
col4.metric("Total Records", f"{total_records:,}")

st.caption(
    f"Date range: {filtered_df[datetime_col].min()} to {filtered_df[datetime_col].max()}"
)

st.divider()


# -----------------------------
# PM2.5 trend chart
# -----------------------------
st.subheader("PM2.5 Concentration Trend Over Time")

trend_df = filtered_df[[datetime_col, pm25_col]].dropna().sort_values(datetime_col)

fig_pm25 = px.line(
    trend_df,
    x=datetime_col,
    y=pm25_col,
    title="PM2.5 Concentration Trend"
)

fig_pm25.update_layout(
    xaxis_title="Datetime",
    yaxis_title="PM2.5"
)

st.plotly_chart(fig_pm25, use_container_width=True)


# -----------------------------
# Other pollutant / weather charts
# -----------------------------
st.subheader("Additional Air Quality and Weather Trends")

possible_extra_cols = [
    "pm10_concentration_ug_m_3",
    "so2_concentration_ug_m_3",
    "no2_concentration_ug_m_3",
    "co_concentration_ug_m_3",
    "o3_concentration_ug_m_3",
    "temperature_fahrenheit",
    "humidity",
    "pressure_hpa",
    "wind_speed_m_s",
]

available_extra_cols = [col for col in possible_extra_cols if col in filtered_df.columns]

if available_extra_cols:
    selected_extra_col = st.selectbox(
        "Select another variable to visualize",
        available_extra_cols
    )

    extra_df = filtered_df[[datetime_col, selected_extra_col]].dropna().sort_values(datetime_col)

    fig_extra = px.line(
        extra_df,
        x=datetime_col,
        y=selected_extra_col,
        title=f"{selected_extra_col} Trend"
    )

    fig_extra.update_layout(
        xaxis_title="Datetime",
        yaxis_title=selected_extra_col
    )

    st.plotly_chart(fig_extra, use_container_width=True)
else:
    st.info("No extra pollutant/weather columns found for additional charts.")

st.divider()


# -----------------------------
# Forecast results
# -----------------------------
st.subheader("LSTM Forecast Results")

pred_df = load_predictions()

if pred_df is None:
    st.warning(
        "Forecast prediction file not found.\n\n"
        "Expected file: reports/validation_predictions.csv"
    )
else:
    required_prediction_cols = [
        "datetime",
        "actual_pm2_5",
        "predicted_pm2_5"
    ]

    missing_cols = [col for col in required_prediction_cols if col not in pred_df.columns]

    if missing_cols:
        st.warning(f"Prediction file found, but missing columns: {missing_cols}")
        st.dataframe(pred_df.head(20), use_container_width=True)
    else:
        forecast_fig = px.line(
            pred_df,
            x="datetime",
            y=["actual_pm2_5", "predicted_pm2_5"],
            title="Actual vs Predicted PM2.5"
        )

        forecast_fig.update_layout(
            xaxis_title="Datetime",
            yaxis_title="PM2.5",
            legend_title="Series"
        )

        st.plotly_chart(forecast_fig, use_container_width=True)

        st.write("Forecast prediction sample:")
        st.dataframe(pred_df.head(50), use_container_width=True)


# -----------------------------
# Model metrics
# -----------------------------
# -----------------------------
# Model metrics
# -----------------------------
st.subheader("Model Performance")

metrics = load_model_metrics()

if metrics is None:
    st.info(
        "Model metrics file not found or unreadable.\n\n"
        "Expected file: reports/training_metrics.json"
    )
else:
    final_metrics = metrics.get("final_metrics", {})

    val_mae = final_metrics.get("val_mae_pm25")
    val_rmse = final_metrics.get("val_rmse_pm25")
    train_loss = final_metrics.get("train_loss")
    val_loss = final_metrics.get("val_loss")

    metric_cols = st.columns(4)

    if val_mae is not None:
        metric_cols[0].metric("Validation MAE", f"{val_mae:.2f}")

    if val_rmse is not None:
        metric_cols[1].metric("Validation RMSE", f"{val_rmse:.2f}")

    if train_loss is not None:
        metric_cols[2].metric("Train Loss", f"{train_loss:.6f}")

    if val_loss is not None:
        metric_cols[3].metric("Validation Loss", f"{val_loss:.6f}")

    st.write("### Training Configuration")

    config_cols = st.columns(4)

    config_cols[0].metric("Model", "LSTM")
    config_cols[1].metric("Epochs", metrics.get("epochs", "N/A"))
    config_cols[2].metric("Sequence Length", metrics.get("sequence_length", "N/A"))
    config_cols[3].metric("Batch Size", metrics.get("batch_size", "N/A"))

    with st.expander("View full training metrics JSON"):
        st.json(metrics)