import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Load data
df = pd.read_csv('/kaggle/input/price-pred/madanapalli_tom_prices.csv')

# Let's assume the relevant columns are 'date' and 'commodity_price'
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df[['Date', 'Modal Price (Rs./Quintal)']]

# Sort data by date
df = df.sort_values(by='Date')

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(df['Modal Price (Rs./Quintal)'].values.reshape(-1, 1))

# Create sequences for LSTM input
seq_length = 30  # Example sequence length

class CommodityPriceDataset(Dataset):
    def _init_(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def _len_(self):
        return len(self.data) - self.seq_length
    
    def _getitem_(self, idx):
        seq = self.data[idx:idx + self.seq_length]
        label = self.data[idx + self.seq_length]
        return torch.tensor(seq, dtype=torch.float).reshape(-1, 1), torch.tensor(label, dtype=torch.float)

# Create training and test datasets
train_size = int(len(scaled_prices) * 0.8)
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

train_dataset = CommodityPriceDataset(train_data, seq_length)
test_dataset = CommodityPriceDataset(test_data, seq_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM model
class LSTMModel(nn.Module):
    def _init_(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self)._init_()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last output
        return out

# Initialize the model
model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.float()
        targets = targets.float()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation function
def evaluate_model(predictions, actuals):
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mae, rmse, mape

# Collect predictions and actual values
test_predictions = []
test_actuals = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        inputs = inputs.float()  # Ensure correct type
        outputs = model(inputs)  # Forward pass
        test_predictions.extend(outputs.numpy().flatten())
        test_actuals.extend(targets.numpy().flatten())

# Reverse scaling for predictions and actual values
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
test_actuals = scaler.inverse_transform(np.array(test_actuals).reshape(-1, 1)).flatten()

# Evaluate the model
mae, rmse, mape = evaluate_model(test_predictions, test_actuals)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_actuals, label="Actual Prices", marker="o")
plt.plot(test_predictions, label="Predicted Prices", marker="x")
plt.title("Actual vs Predicted Commodity Prices")
plt.xlabel("Time Steps")
plt.ylabel("Price (Rs./Quintal)")
plt.legend()
plt.grid()
plt.show()

# Forecast future prices (next 10 steps)
def make_forecast(model, data, seq_length=30, steps=10):
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # Prepare the input sequence
    input_seq = torch.tensor(data[-seq_length:], dtype=torch.float32).view(1, seq_length, 1).to(device)
    forecast = []
    
    with torch.no_grad():
        for _ in range(steps):
            # Predict the next value
            pred = model(input_seq)
            forecast.append(pred.item())
            
            # Update the input sequence
            # Convert pred to match input_seq dimensions
            pred = pred.view(1, 1, 1)  # Shape (1, 1, 1)
            input_seq = torch.cat((input_seq[:, 1:, :], pred), dim=1)
    
    return forecast

# Forecast next 10 commodity prices
future_forecast = make_forecast(model, scaled_prices, seq_length=seq_length, steps=10)
future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten()

# Print future forecasted prices
print("Future Commodity Price Predictions (Next 10 Days):")
for i, price in enumerate(future_forecast, 1):
    print(f"Day {i}: {price:.2f}")