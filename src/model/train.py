import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.model.dataset import (
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMN,
    load_hourly_data,
)
from src.model.lstm_model import AirQualityLSTM


MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_PATH = MODEL_DIR / "airman_lstm.pt"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.pkl"
METRICS_PATH = REPORTS_DIR / "training_metrics.json"

TRAIN_RATIO = 0.8
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_windows(features, target, sequence_length):
    x_values = []
    y_values = []

    for i in range(len(features) - sequence_length):
        x = features[i : i + sequence_length]
        y = target[i + sequence_length]

        x_values.append(x)
        y_values.append(y)

    return np.array(x_values), np.array(y_values).reshape(-1)


def make_dataloader(x, y, batch_size, shuffle):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate(model, dataloader, criterion, target_scaler, device):
    model.eval()

    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item() * x_batch.size(0)

            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    predictions_real = target_scaler.inverse_transform(predictions).reshape(-1)
    actuals_real = target_scaler.inverse_transform(actuals).reshape(-1)

    mae = mean_absolute_error(actuals_real, predictions_real)
    rmse = np.sqrt(mean_squared_error(actuals_real, predictions_real))

    return {
        "loss": float(avg_loss),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def main():
    set_seed()

    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hourly_df = load_hourly_data()

    print("Hourly data loaded.")
    print(f"Shape: {hourly_df.shape}")
    print(f"Date range: {hourly_df['datetime'].min()} to {hourly_df['datetime'].max()}")

    split_index = int(len(hourly_df) * TRAIN_RATIO)

    train_df = hourly_df.iloc[:split_index].copy()
    val_df = hourly_df.iloc[split_index:].copy()

    print("\nTrain/validation split:")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Train date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
    print(f"Validation date range: {val_df['datetime'].min()} to {val_df['datetime'].max()}")

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features = feature_scaler.fit_transform(train_df[FEATURE_COLUMNS])
    train_target = target_scaler.fit_transform(train_df[[TARGET_COLUMN]])

    val_features = feature_scaler.transform(val_df[FEATURE_COLUMNS])
    val_target = target_scaler.transform(val_df[[TARGET_COLUMN]])

    x_train, y_train = create_windows(train_features, train_target, SEQUENCE_LENGTH)
    x_val, y_val = create_windows(val_features, val_target, SEQUENCE_LENGTH)

    print("\nSliding-window shapes:")
    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_val: {x_val.shape}")
    print(f"y_val: {y_val.shape}")

    train_loader = make_dataloader(x_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, BATCH_SIZE, shuffle=False)

    model = AirQualityLSTM(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []

    print("\nStarting training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_total = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * x_batch.size(0)

        train_loss = train_loss_total / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, target_scaler, device)

        epoch_result = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": val_metrics["loss"],
            "val_mae_pm25": val_metrics["mae"],
            "val_rmse_pm25": val_metrics["rmse"],
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val MAE: {val_metrics['mae']:.2f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f}"
        )

    final_metrics = history[-1]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": len(FEATURE_COLUMNS),
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            },
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "sequence_length": SEQUENCE_LENGTH,
        },
        MODEL_PATH,
    )

    with open(FEATURE_SCALER_PATH, "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(TARGET_SCALER_PATH, "wb") as f:
        pickle.dump(target_scaler, f)

    metrics_report = {
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": SEQUENCE_LENGTH,
        "train_ratio": TRAIN_RATIO,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "train_date_start": str(train_df["datetime"].min()),
        "train_date_end": str(train_df["datetime"].max()),
        "validation_date_start": str(val_df["datetime"].min()),
        "validation_date_end": str(val_df["datetime"].max()),
        "final_metrics": final_metrics,
        "history": history,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Feature scaler saved to: {FEATURE_SCALER_PATH}")
    print(f"Target scaler saved to: {TARGET_SCALER_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()