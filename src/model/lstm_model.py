import torch
import torch.nn as nn


class AirQualityLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        # Use the last time step output for prediction
        last_time_step = lstm_output[:, -1, :]

        prediction = self.fc(last_time_step)

        return prediction.squeeze(-1)


def main():
    batch_size = 32
    sequence_length = 24
    input_size = 12

    model = AirQualityLSTM(input_size=input_size)

    dummy_input = torch.randn(batch_size, sequence_length, input_size)
    dummy_output = model(dummy_input)

    print("LSTM model created successfully.")
    print(model)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")


if __name__ == "__main__":
    main()