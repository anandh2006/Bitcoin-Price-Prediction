import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Step 1: Load the dataset
df = yf.download('BTC-USD', start='2020-01-01', end='2024-01-01')
df = df[['Close']]  # Keeping only 'Close' price for prediction
df.dropna(inplace=True)

# Step 2: Data Preprocessing for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Step 3: Create Training and Testing Data for LSTM
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 60  # 60-day window
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 4: Build the LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Make Predictions with LSTM
lstm_predictions = model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Step 6: ARIMA Model
train_arima, test_arima = df[:train_size], df[train_size:]

# Find optimal ARIMA parameters using auto_arima
auto_arima_model = pm.auto_arima(train_arima, seasonal=False, stepwise=True, suppress_warnings=True)
print(f"Best ARIMA Model: {auto_arima_model.order}")

# Train ARIMA model with best parameters
arima_model = ARIMA(train_arima, order=auto_arima_model.order)
arima_model_fit = arima_model.fit()

# Make Predictions with ARIMA
arima_forecast = arima_model_fit.forecast(steps=len(test_arima))

# Step 7: Evaluate Models

# Trim y_test_actual to match LSTM predictions
y_test_actual = test_arima.values
y_test_actual_trimmed = y_test_actual[-len(lstm_predictions):]

lstm_mae = mean_absolute_error(y_test_actual_trimmed, lstm_predictions)
lstm_rmse = np.sqrt(mean_squared_error(y_test_actual_trimmed, lstm_predictions))

arima_mae = mean_absolute_error(y_test_actual, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(y_test_actual, arima_forecast))

print(f'LSTM MAE: {lstm_mae}, LSTM RMSE: {lstm_rmse}')
print(f'ARIMA MAE: {arima_mae}, ARIMA RMSE: {arima_rmse}')

# Step 8: Plot Predictions vs Actual
plt.figure(figsize=(14, 6))
plt.plot(df.index[train_size:], y_test_actual, label="Actual Price", color='blue')
plt.plot(df.index[train_size+sequence_length:], lstm_predictions, label="LSTM Predictions", color='red')
plt.plot(df.index[train_size:], arima_forecast, label="ARIMA Predictions", color='green')
plt.xlabel("Date")
plt.ylabel("Bitcoin Price (USD)")
plt.title("Bitcoin Price Prediction: LSTM vs ARIMA")
plt.legend()
plt.show()
