import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load stock data (already available data)
stock_data = pd.read_csv('../Data/reliance.csv')

# Convert the 'Date' column to datetime
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close']].values)

# Create dataset with time steps
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, Y_train = create_dataset(scaled_data[:int(len(scaled_data)*0.8)], time_step)
X_test, Y_test = create_dataset(scaled_data[int(len(scaled_data)*0.8):], time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and train the model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, Y_train, epochs=50, batch_size=64)

# Predict future stock prices
def predict_future(model, recent_data, steps=30):
    predictions = []
    current_input = recent_data[-time_step:].reshape(1, time_step, 1)  # Start with the last available input
    
    for _ in range(steps):
        next_pred = model.predict(current_input)
        predictions.append(next_pred[0, 0])  # Append the prediction
        
        # Update input with the new prediction (reshaping next_pred to be compatible)
        current_input = np.append(current_input[:, 1:, :], [[[next_pred[0, 0]]]], axis=1)
    
    return np.array(predictions)

# Get the last data points to predict future values
recent_data = scaled_data[-time_step:]

# Predict the next 30 days of stock prices
future_predictions = predict_future(model, recent_data, steps=30)

# Inverse scale the predicted data
future_predictions_scaled = scaler.inverse_transform(future_predictions.reshape(-1, 1))

# Generate future dates
last_date = stock_data['Date'].values[-1]  # Get the last date from the dataset
future_dates = pd.date_range(last_date, periods=30, freq='B')  # Generate 30 future business days

# Print or plot the future predictions
# Print the future dates along with their predicted stock prices
for date, price in zip(future_dates, future_predictions_scaled):
    print(f"Date: {date.date()}, Predicted Price: {price[0]:.2f}")


# Visualize the results
plt.figure(figsize=(14,7))
plt.plot(stock_data['Date'], stock_data['Close'], color='blue', label='Historical Stock Price')
plt.plot(future_dates, future_predictions_scaled, color='red', label='Predicted Future Stock Price')
plt.title('Stock Price Prediction for Future Dates')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.xticks(rotation=45)
plt.legend()
plt.show()