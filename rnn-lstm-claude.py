import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

class TimeSeriesDataset(Dataset):
    """Custom dataset for multivariate time series data"""
    
    def __init__(self, data, seq_length, target_col=0):
        self.data = data
        self.seq_length = seq_length
        self.target_col = target_col
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Input sequence: all features for seq_length timesteps
        x = self.data[idx:idx + self.seq_length]
        # Target: next value of target column
        y = self.data[idx + self.seq_length, self.target_col]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class SimpleRNN(nn.Module):
    """Simple RNN for multivariate time series forecasting"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Take output from last time step
        out = out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class LSTMModel(nn.Module):
    """LSTM model for multivariate time series forecasting"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take output from last time step
        out = out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TimeSeriesTrainer:
    """Trainer class for time series models"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader, epochs, lr=0.001, patience=10):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    def predict(self, data_loader):
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        return np.array(predictions), np.array(actuals)
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def create_sample_data(n_samples=1000, n_features=3):
    """Create sample multivariate time series data"""
    np.random.seed(42)
    
    # Generate time series with trend, seasonality, and noise
    t = np.linspace(0, 10, n_samples)
    
    # Feature 1: trend + seasonality
    feature1 = 0.5 * t + 2 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, n_samples)
    
    # Feature 2: different seasonality pattern
    feature2 = np.cos(np.pi * t) + 0.3 * t + np.random.normal(0, 0.1, n_samples)
    
    # Feature 3: target variable influenced by other features
    feature3 = 0.3 * feature1 + 0.2 * feature2 + np.sin(4 * np.pi * t) + np.random.normal(0, 0.1, n_samples)
    
    data = np.column_stack([feature3, feature1, feature2])  # target is first column
    return data

def prepare_data(data, seq_length=20, train_ratio=0.7, val_ratio=0.2):
    """Prepare data for training"""
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Split data
    n_train = int(len(data_scaled) * train_ratio)
    n_val = int(len(data_scaled) * val_ratio)
    
    train_data = data_scaled[:n_train]
    val_data = data_scaled[n_train:n_train + n_val]
    test_data = data_scaled[n_train + n_val:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, seq_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length)
    
    return train_dataset, val_dataset, test_dataset, scaler

def evaluate_model(predictions, actuals, scaler, target_col=0):
    """Evaluate model performance"""
    # Inverse transform predictions and actuals
    pred_full = np.zeros((len(predictions), scaler.n_features_in_))
    actual_full = np.zeros((len(actuals), scaler.n_features_in_))
    
    pred_full[:, target_col] = predictions.flatten()
    actual_full[:, target_col] = actuals.flatten()
    
    pred_inverse = scaler.inverse_transform(pred_full)[:, target_col]
    actual_inverse = scaler.inverse_transform(actual_full)[:, target_col]
    
    # Calculate metrics
    mse = mean_squared_error(actual_inverse, pred_inverse)
    mae = mean_absolute_error(actual_inverse, pred_inverse)
    rmse = np.sqrt(mse)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'predictions': pred_inverse,
        'actuals': actual_inverse
    }

def main():
    """Main function to demonstrate RNN and LSTM models"""
    
    # Parameters
    seq_length = 20
    hidden_size = 64
    num_layers = 2
    batch_size = 32
    epochs = 100
    
    # Create sample data
    print("Creating sample multivariate time series data...")
    data = create_sample_data(n_samples=1000, n_features=3)
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, scaler = prepare_data(data, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = data.shape[1]  # number of features
    output_size = 1  # predicting one target variable
    
    # Train RNN
    print("\nTraining Simple RNN...")
    rnn_model = SimpleRNN(input_size, hidden_size, num_layers, output_size)
    rnn_trainer = TimeSeriesTrainer(rnn_model)
    rnn_trainer.train(train_loader, val_loader, epochs)
    
    # Train LSTM
    print("\nTraining LSTM...")
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    lstm_trainer = TimeSeriesTrainer(lstm_model)
    lstm_trainer.train(train_loader, val_loader, epochs)
    
    # Evaluate models
    print("\nEvaluating models...")
    
    # RNN evaluation
    rnn_pred, rnn_actual = rnn_trainer.predict(test_loader)
    rnn_results = evaluate_model(rnn_pred, rnn_actual, scaler)
    
    # LSTM evaluation
    lstm_pred, lstm_actual = lstm_trainer.predict(test_loader)
    lstm_results = evaluate_model(lstm_pred, lstm_actual, scaler)
    
    # Print results
    print("\nRNN Results:")
    print(f"MSE: {rnn_results['MSE']:.6f}")
    print(f"MAE: {rnn_results['MAE']:.6f}")
    print(f"RMSE: {rnn_results['RMSE']:.6f}")
    
    print("\nLSTM Results:")
    print(f"MSE: {lstm_results['MSE']:.6f}")
    print(f"MAE: {lstm_results['MAE']:.6f}")
    print(f"RMSE: {lstm_results['RMSE']:.6f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot training losses
    plt.subplot(2, 2, 1)
    plt.plot(rnn_trainer.train_losses, label='RNN Train')
    plt.plot(rnn_trainer.val_losses, label='RNN Val')
    plt.title('RNN Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(lstm_trainer.train_losses, label='LSTM Train')
    plt.plot(lstm_trainer.val_losses, label='LSTM Val')
    plt.title('LSTM Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot predictions
    plt.subplot(2, 2, 3)
    plt.plot(rnn_results['actuals'][:100], label='Actual', alpha=0.7)
    plt.plot(rnn_results['predictions'][:100], label='RNN Prediction', alpha=0.7)
    plt.title('RNN Predictions vs Actual (First 100 points)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(lstm_results['actuals'][:100], label='Actual', alpha=0.7)
    plt.plot(lstm_results['predictions'][:100], label='LSTM Prediction', alpha=0.7)
    plt.title('LSTM Predictions vs Actual (First 100 points)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()