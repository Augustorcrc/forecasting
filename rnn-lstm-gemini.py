# Full Script for Multivariate Time Series Forecasting with RNN and LSTM

# ---------------------------------
# Step 0: Import Libraries
# ---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense

print(f"TensorFlow Version: {tf.__version__}")

# ---------------------------------
# Step 1: Data Preparation
# ---------------------------------

# --- 1.1: Generate Synthetic Data ---
# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("\n--- Step 1: Data Preparation ---")
# Generate synthetic time series data
time = np.arange(0, 500, 1)
# Feature 1: Sine wave with noise (our target)
feature_1 = np.sin(time * 0.1) + np.random.normal(0, 0.05, len(time))
# Feature 2: Linear trend
feature_2 = 0.01 * time
# Feature 3: Another periodic signal with more noise
feature_3 = 0.5 * np.sin(time * 0.2) + np.random.normal(0, 0.1, len(time))

# Create a pandas DataFrame
data = pd.DataFrame({
    'Feature_1': feature_1,
    'Feature_2': feature_2,
    'Feature_3': feature_3
})

print("Original Data Head:")
print(data.head())

# Visualize the raw data
data.plot(subplots=True, figsize=(10, 8), title="Synthetic Multivariate Time Series")
plt.suptitle("Raw Input Data")
plt.show()


# --- 1.2: Scale Data ---
# Neural networks perform best when data is scaled to a small range.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
print("\nScaled Data Head:")
print(scaled_data[:5])


# --- 1.3: Create Sequences ---
def create_sequences(data, n_steps, target_col_index):
    """
    Creates sequences for a multivariate time series.
    'data': The scaled dataset.
    'n_steps': The number of past timesteps to use for prediction.
    'target_col_index': The index of the column we want to predict.
    """
    X, y = [], []
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(data)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, target_col_index]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Hyperparameters
n_steps = 30 # Use 30 past steps to predict the next step
n_features = data.shape[1] # We have 3 features
target_column_index = 0 # We want to predict 'Feature_1'

# Generate the sequences
X, y = create_sequences(scaled_data, n_steps, target_column_index)

print(f"\nShape of X (input sequences): {X.shape}") # Expected: (samples, n_steps, n_features)
print(f"Shape of y (target values): {y.shape}")   # Expected: (samples,)


# --- 1.4: Train-Test Split ---
# For time series, the split must be chronological.
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("\nData Split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# ---------------------------------
# Step 2: RNN Model Implementation
# ---------------------------------
print("\n--- Step 2: RNN Model ---")
# Define the RNN model
rnn_model = Sequential([
    Input(shape=(n_steps, n_features)),
    SimpleRNN(50, activation='relu'), # 50 RNN units
    Dense(1) # Output layer with 1 neuron for regression
])

# Compile the model
rnn_model.compile(optimizer='adam', loss='mean_squared_error')
print("RNN Model Summary:")
rnn_model.summary()

# Train the model
print("\nTraining RNN Model...")
rnn_history = rnn_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1, # Use 10% of training data for validation
    verbose=1
)


# ---------------------------------
# Step 3: LSTM Model Implementation
# ---------------------------------
print("\n--- Step 3: LSTM Model ---")
# Define the LSTM model
lstm_model = Sequential([
    Input(shape=(n_steps, n_features)),
    LSTM(50, activation='relu'), # 50 LSTM units
    Dense(1) # Output layer
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
print("LSTM Model Summary:")
lstm_model.summary()

# Train the model
print("\nTraining LSTM Model...")
lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# ---------------------------------
# Step 4: Evaluation and Comparison
# ---------------------------------
print("\n--- Step 4: Evaluation and Comparison ---")

# --- 4.1: Make Predictions ---
print("Making predictions on the test set...")
rnn_predictions_scaled = rnn_model.predict(X_test)
lstm_predictions_scaled = lstm_model.predict(X_test)


# --- 4.2: Inverse Transform Predictions ---
# The predictions are on the scaled scale [0, 1]. We need to inverse
# transform them to compare with the original data.
# We create a dummy array of the correct shape and put our predictions
# in the target column to perform inverse scaling.

# Inverse transform RNN predictions
dummy_rnn = np.zeros((len(rnn_predictions_scaled), n_features))
dummy_rnn[:, target_column_index] = rnn_predictions_scaled.ravel()
rnn_predictions = scaler.inverse_transform(dummy_rnn)[:, target_column_index]

# Inverse transform LSTM predictions
dummy_lstm = np.zeros((len(lstm_predictions_scaled), n_features))
dummy_lstm[:, target_column_index] = lstm_predictions_scaled.ravel()
lstm_predictions = scaler.inverse_transform(dummy_lstm)[:, target_column_index]

# Inverse transform the actual test labels (y_test) for comparison
dummy_y_test = np.zeros((len(y_test), n_features))
dummy_y_test[:, target_column_index] = y_test.ravel()
y_test_actual = scaler.inverse_transform(dummy_y_test)[:, target_column_index]


# --- 4.3: Visualize Results ---
print("Visualizing results...")
plt.figure(figsize=(15, 7))
plt.plot(y_test_actual, label='Actual Values', color='blue', marker='.')
plt.plot(rnn_predictions, label='RNN Predictions', color='orange', linestyle='--')
plt.plot(lstm_predictions, label='LSTM Predictions', color='green', linestyle='--')
plt.title('Multivariate Time Series Forecasting: Actual vs. Predicted')
plt.xlabel('Time Step (in test set)')
plt.ylabel(f'Value of {data.columns[target_column_index]}')
plt.legend()
plt.grid(True)
plt.show()


# --- 4.4: Calculate Metrics ---
rnn_mse = mean_squared_error(y_test_actual, rnn_predictions)
lstm_mse = mean_squared_error(y_test_actual, lstm_predictions)
rnn_rmse = np.sqrt(rnn_mse)
lstm_rmse = np.sqrt(lstm_mse)

print("\n--- Model Performance Metrics ---")
print(f"Simple RNN MSE: {rnn_mse:.6f}, RMSE: {rnn_rmse:.6f}")
print(f"LSTM MSE:       {lstm_mse:.6f}, RMSE: {lstm_rmse:.6f}")

if lstm_mse < rnn_mse:
    print("\nConclusion: LSTM performed better than Simple RNN on this task.")
else:
    print("\nConclusion: Simple RNN performed better than or equal to LSTM on this task.")