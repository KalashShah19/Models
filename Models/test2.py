import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Function to calculate RSI (Relative Strength Index)
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

# Function to calculate MACD (Moving Average Convergence Divergence)
def calculate_macd(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Step 1: Load stock data (assuming you have Reliance stock data)
stock_data = pd.read_csv('../Data/reliance.csv')

# Step 2: Add technical indicators (SMA, RSI, MACD, Volume)
stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
stock_data['RSI'] = calculate_rsi(stock_data['Close'], window=14)
stock_data['MACD'], stock_data['Signal_Line'] = calculate_macd(stock_data['Close'])

# Lag features: Add past technical indicators as additional features
for lag in [1, 2, 3, 5]:
    stock_data[f'RSI_Lag_{lag}'] = stock_data['RSI'].shift(lag)
    stock_data[f'MACD_Lag_{lag}'] = stock_data['MACD'].shift(lag)
    stock_data[f'SMA50_Lag_{lag}'] = stock_data['SMA50'].shift(lag)

# Drop NaN values created by rolling and lag features
stock_data.dropna(inplace=True)

# Step 3: Preprocess data by scaling individual features separately
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_sma50 = MinMaxScaler(feature_range=(0, 1))
scaler_sma200 = MinMaxScaler(feature_range=(0, 1))
scaler_rsi = MinMaxScaler(feature_range=(0, 1))
scaler_macd = MinMaxScaler(feature_range=(0, 1))

scaled_close = scaler_close.fit_transform(stock_data[['Close']])
scaled_sma50 = scaler_sma50.fit_transform(stock_data[['SMA50']])
scaled_sma200 = scaler_sma200.fit_transform(stock_data[['SMA200']])
scaled_rsi = scaler_rsi.fit_transform(stock_data[['RSI']])
scaled_macd = scaler_macd.fit_transform(stock_data[['MACD']])

# Combine scaled features including lag features into a single dataset
scaled_data = np.concatenate((
    scaled_close, 
    scaled_sma50, 
    scaled_sma200, 
    scaled_rsi, 
    scaled_macd, 
    stock_data[['RSI_Lag_1', 'RSI_Lag_2', 'RSI_Lag_3', 'RSI_Lag_5']].values, 
    stock_data[['MACD_Lag_1', 'MACD_Lag_2', 'MACD_Lag_3', 'MACD_Lag_5']].values
), axis=1)

# Step 4: Split into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Create dataset with time steps
def create_dataset(data, time_step=120):  # Increased time step to 120 days
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])  # Use all features (Close, SMA, RSI, MACD, lag features)
        Y.append(data[i, 0])  # Predicting the next close price
    return np.array(X), np.array(Y)

time_step = 120  # Increased time step for longer trend learning
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# Step 5: Reshape input for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Step 6: Build a deeper Bidirectional LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
model.add(Dropout(0.4))
model.add(Dense(units=1))  # Predict the next closing price
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 7: Train the model with Early Stopping
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X_train, Y_train, epochs=150, batch_size=32, callbacks=[early_stopping])

# Step 8: Make predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler_close.inverse_transform(predicted_stock_price)
real_stock_price = scaler_close.inverse_transform(Y_test.reshape(-1, 1))

# Step 9: Calculate directional accuracy based on stock movement over a window of 5 days
def calculate_movement(prices, window=5):
    return np.where(np.diff(prices.flatten(), n=window) > 0, 1, 0)

real_movement = calculate_movement(real_stock_price, window=5)
predicted_movement = calculate_movement(predicted_stock_price, window=5)

accuracy = accuracy_score(real_movement, predicted_movement)

# Step 10: Calculate and print RMSE and accuracy
rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

print(f"RMSE: {rmse}")
print(f"Directional Accuracy Score: {accuracy}")

# Step 11: Plot the results
plt.figure(figsize=(14,7))
plt.plot(real_stock_price, color='blue', label='Real Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
plt.title('Reliance Stock Price Prediction with Bidirectional LSTM and Lag Features')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
