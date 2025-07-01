
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Generación de datos
np.random.seed(42)
n = 365
temperatura = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.normal(0, 1, n)
publicidad = np.random.randint(0, 2, size=n)
dia_semana = np.tile(np.arange(7), n // 7 + 1)[:n]
demanda = (
    100 + 5 * temperatura +
    20 * publicidad -
    10 * (dia_semana == 6) - 15 * (dia_semana == 0) +
    np.random.normal(0, 5, n)
)

df = pd.DataFrame({
    'temperatura': temperatura,
    'publicidad': publicidad,
    'dia_semana': dia_semana,
    'demanda': demanda
})

# 2. Preprocesamiento
features = ['temperatura', 'publicidad', 'dia_semana']
target = ['demanda']
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(df[features])
y = scaler_y.fit_transform(df[target])

# 3. Dataset para series temporales
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size):
        self.X = []
        self.y = []
        for i in range(len(X) - window_size):
            self.X.append(X[i:i+window_size])
            self.y.append(y[i+window_size])
        self.X = torch.tensor(self.X).float()
        self.y = torch.tensor(self.y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

window_size = 14
dataset = TimeSeriesDataset(X, y, window_size)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Modelos
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 5. Entrenamiento
def train(model, loader, epochs=30):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        loss_total = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_total/len(loader):.4f}")

# 6. Instanciar y entrenar modelos
input_size = len(features)
rnn_model = RNNModel(input_size)
lstm_model = LSTMModel(input_size)

print("Entrenando RNN...")
train(rnn_model, loader)

print("\nEntrenando LSTM...")
train(lstm_model, loader)
