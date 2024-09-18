import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

sys.stdout.reconfigure(encoding='utf-8')

# Step 1: Load the stock data
df = pd.read_csv('D:/! Kalash/AI/Models/Stocks/Data/reliance.csv')  # You can replace this with live data fetching
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 2: Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

time_step = 60  # You can experiment with this value
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# Reshape data to [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 3: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the model
model.fit(X_train, Y_train, epochs=50, batch_size=64)

# Step 5: Make predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
real_stock_price = scaler.inverse_transform(test_data[time_step:])

# Step 7: Plot the results
plt.plot(df.index[train_size + time_step:], scaler.inverse_transform(test_data[time_step:]), label='Real Stock Price')
plt.plot(df.index[train_size + time_step:], predicted_stock_price, label='Predicted Stock Price')
plt.title('Reliance Price Prediction')
plt.legend()
plt.show()


# Step 8: Calculate Accuracy Score for Stock Movement Prediction

# Convert real and predicted stock prices to binary movement (1 = price increased, 0 = price decreased)
real_movement = np.where(np.diff(real_stock_price.flatten()) > 0, 1, 0)
predicted_movement = np.where(np.diff(predicted_stock_price.flatten()) > 0, 1, 0)

# Calculate accuracy score
accuracy = accuracy_score(real_movement, predicted_movement)

# Step 7: Calculate and print RMSE and accuracy
rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

print(f"RMSE: {rmse}")
print(f"Accuracy Score: {accuracy}")