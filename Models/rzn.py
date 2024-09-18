import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import shap

# Function to calculate the RSI (Relative Strength Index)
def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Step 1: Load stock data
stock_data = pd.read_csv('../Data/reliance.csv') 

# Step 2: Add technical indicators (SMA, RSI)
stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
stock_data['RSI'] = calculate_rsi(stock_data['Close'], window=14)
stock_data.dropna(inplace=True)

# Step 3: Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close', 'SMA50', 'SMA200', 'RSI']].values)

# Create dataset with time steps
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])
        Y.append(data[i, 0])  # Predicting the next close price
    return np.array(X), np.array(Y)

time_step = 60
train_size = int(len(scaled_data) * 0.8)
X_train, Y_train = create_dataset(scaled_data[:train_size], time_step)
X_test, Y_test = create_dataset(scaled_data[train_size:], time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Step 4: Build the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
model.fit(X_train, Y_train, epochs=50, batch_size=64)

# Predict future stock prices
def predict_future(model, recent_data, steps=30):
    predictions = []
    current_input = recent_data[-time_step:].reshape(1, time_step, recent_data.shape[1])
    
    for _ in range(steps):
        next_pred = model.predict(current_input)
        predictions.append(next_pred[0, 0])  # Append the predicted price
        
        # Update input with the new prediction
        current_input = np.append(current_input[:, 1:, :], [[[next_pred[0, 0], recent_data[-1, 1], recent_data[-1, 2], recent_data[-1, 3]]]], axis=1)
    
    return np.array(predictions)

# Step 6: Get the last data points to predict future values
recent_data = scaled_data[-time_step:]

# Predict the next 30 days of stock prices
future_predictions = predict_future(model, recent_data, steps=30)

# Inverse scale the predicted data
future_predictions_scaled = scaler.inverse_transform(np.concatenate((future_predictions.reshape(-1, 1), np.zeros((future_predictions.shape[0], 3))), axis=1))[:, 0]

# Step 7: Explain predictions based on input features (SMA, RSI, etc.)
def explain_prediction(recent_data):
    close_price = recent_data[:, 0]
    sma50 = recent_data[:, 1]
    sma200 = recent_data[:, 2]
    rsi = recent_data[:, 3]
    
    explanation = []
    if close_price[-1] > sma50[-1]:
        explanation.append("The price is above the 50-day SMA, indicating an upward trend.")
    else:
        explanation.append("The price is below the 50-day SMA, indicating a downward trend.")
    
    if close_price[-1] > sma200[-1]:
        explanation.append("The price is above the 200-day SMA, indicating long-term strength.")
    else:
        explanation.append("The price is below the 200-day SMA, indicating long-term weakness.")
    
    if rsi[-1] > 70:
        explanation.append("RSI indicates the stock is overbought.")
    elif rsi[-1] < 30:
        explanation.append("RSI indicates the stock is oversold.")
    else:
        explanation.append("RSI is neutral.")
    
    return "\n".join(explanation)

# Print the reasoning for the predictions
reasoning = explain_prediction(recent_data)
print("Reasoning for the prediction:")
print(reasoning)

# Step 8: Visualize the results
plt.figure(figsize=(14,7))
plt.plot(stock_data['Close'], color='blue', label='Historical Stock Price')
plt.plot(range(len(stock_data), len(stock_data) + 30), future_predictions_scaled, color='red', label='Predicted Future Stock Price')
plt.title('Stock Price Prediction for Future Dates')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Step 9: Save the predicted prices and real prices to a CSV file
output_df = pd.DataFrame({
    'Predicted Future Stock Price': future_predictions_scaled
})

# Save to CSV
output_df.to_csv('predicted_stock_prices_with_reasoning.csv', index=False)
print("Predicted stock prices saved to 'predicted_stock_prices_with_reasoning.csv'.")

