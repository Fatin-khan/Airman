import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from src.model.dataset import (
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMN,
    load_hourly_data,
)
from src.model.lstm_model import AirQualityLSTM


MODEL_PATH = Path("models/airman_lstm.pt")
FEATURE_SCALER_PATH = Path("models/feature_scaler.pkl")
TARGET_SCALER_PATH = Path("models/target_scaler.pkl")
METRICS_PATH = Path("reports/training_metrics.json")
PREDICTIONS_PATH = Path("reports/validation_predictions.csv")

TRAIN_RATIO = 0.8
BATCH_SIZE = 64


def create_windows(features, target, sequence_length):
    x_values = []
    y_values = []

    for i in range(len(features) - sequence_length):
        x = features[i : i + sequence_length]
        y = target[i + sequence_length]

        x_values.append(x)
        y_values.append(y)

    return np.array(x_values), np.array(y_values).reshape(-1)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Train the model first using: python -m src.model.train"
        )

    if not FEATURE_SCALER_PATH.exists():
        raise FileNotFoundError(f"Feature scaler not found: {FEATURE_SCALER_PATH}")

    if not TARGET_SCALER_PATH.exists():
        raise FileNotFoundError(f"Target scaler not found: {TARGET_SCALER_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model_config = checkpoint["model_config"]

    model = AirQualityLSTM(
        input_size=model_config["input_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(FEATURE_SCALER_PATH, "rb") as f:
        feature_scaler = pickle.load(f)

    with open(TARGET_SCALER_PATH, "rb") as f:
        target_scaler = pickle.load(f)

    hourly_df = load_hourly_data()

    split_index = int(len(hourly_df) * TRAIN_RATIO)
    val_df = hourly_df.iloc[split_index:].copy()

    val_features = feature_scaler.transform(val_df[FEATURE_COLUMNS])
    val_target = target_scaler.transform(val_df[[TARGET_COLUMN]])

    x_val, y_val = create_windows(val_features, val_target, SEQUENCE_LENGTH)

    x_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_tensor = torch.tensor(y_val, dtype=torch.float32)

    val_loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    predictions = []
    actuals = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)

            y_pred = model(x_batch)

            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    predictions_real = target_scaler.inverse_transform(predictions).reshape(-1)
    actuals_real = target_scaler.inverse_transform(actuals).reshape(-1)

    mae = mean_absolute_error(actuals_real, predictions_real)
    rmse = np.sqrt(mean_squared_error(actuals_real, predictions_real))

    print("\nEvaluation complete.")
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")

    prediction_datetimes = val_df["datetime"].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)

    results_df = pd.DataFrame(
        {
            "datetime": prediction_datetimes,
            "actual_pm2_5": actuals_real,
            "predicted_pm2_5": predictions_real,
            "absolute_error": np.abs(actuals_real - predictions_real),
        }
    )

    results_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nPredictions saved to: {PREDICTIONS_PATH}")

    print("\nFirst 10 predictions:")
    print(results_df.head(10))

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            training_metrics = json.load(f)

        print("\nTraining file final metrics:")
        print(json.dumps(training_metrics["final_metrics"], indent=4))


if __name__ == "__main__":
    main()