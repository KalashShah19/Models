import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Function to calculate the RSI (Relative Strength Index)
def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)  # Handle division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Step 1: Load and preprocess stock data
stock_data = pd.read_csv('../Data/reliance.csv') 

# Step 2: Add technical indicators (SMA, RSI)
stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
stock_data['RSI'] = calculate_rsi(stock_data['Close'], window=14)
stock_data.dropna(inplace=True)

# Step 3: Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close', 'SMA50', 'SMA200', 'RSI']].values)

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Create dataset with time steps
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])  # Use all features (Close, SMA, RSI)
        Y.append(data[i, 0])  # Predicting the next close price
    return np.array(X), np.array(Y)

time_step = 60  # Use last 60 days of data to predict the next close
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Step 4: Build the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=1))  # Predict the next closing price
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 5: Train the model
model.fit(X_train, Y_train, epochs=50, batch_size=4, validation_split=0.1, callbacks=[early_stopping])

# Step 6: Make predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(np.concatenate((predicted_stock_price, X_test[:, -1, 1:]), axis=1))[:, 0]
real_stock_price = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

# Step 7: Calculate movement (directional) accuracy
def calculate_movement(prices, window=1):
    return np.where(np.diff(prices.flatten(), n=window) > 0, 1, 0)

real_movement = calculate_movement(real_stock_price, window=1)
predicted_movement = calculate_movement(predicted_stock_price, window=1)
directional_accuracy = accuracy_score(real_movement, predicted_movement)

# Step 8: Calculate price accuracy with tolerance
def price_within_tolerance(real_prices, predicted_prices, tolerance=0.02):
    price_diff = np.abs(real_prices - predicted_prices)
    return np.where(price_diff <= (tolerance * real_prices), 1, 0)

tolerance = 0.02  # 2% tolerance for price difference
price_accuracy = price_within_tolerance(real_stock_price, predicted_stock_price, tolerance)

# Step 9: Combine price and movement accuracies
combined_accuracy = np.where((predicted_movement == real_movement) & (price_accuracy == 1), 1, 0)
final_accuracy = np.mean(combined_accuracy)

# Step 10: Calculate and print metrics
rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

print(f"RMSE: {rmse}")
print(f"Directional Accuracy: {directional_accuracy * 100:.2f}%")
print(f"Final Combined Accuracy (Price + Direction): {final_accuracy * 100:.2f}%")

# Step 11: Plot the results
plt.figure(figsize=(14,7))
plt.plot(real_stock_price, color='blue', label='Real Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Step 12: Save the predicted prices and real prices to a CSV file
output_df = pd.DataFrame({
    'Real Stock Price': real_stock_price,
    'Predicted Stock Price': predicted_stock_price
})

# Save to CSV
output_df.to_csv('predicted_stock_prices.csv', index=False)
print("Predicted and real stock prices saved to 'predicted_stock_prices.csv'")