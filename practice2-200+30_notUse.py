import yfinance as yf
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# ============================================================
# 1. Load data
# ============================================================
ticker = "^SPX"
data = yf.download(ticker, start="2015-01-01", end="2025-01-01")

df = data[["Close"]].rename(columns={"Close": "price"})
df = df.dropna()
dates = df.index
n_total = len(df)

print("Total data:", n_total)


# ============================================================
# 2. Split
# ============================================================
n = len(df)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]


# ============================================================
# 3. Scaling
# ============================================================
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
val_scaled   = scaler.transform(val)
test_scaled  = scaler.transform(test)


# ============================================================
# 4. Dataset (1-step forecasting)
# ============================================================
def make_dataset_1step(series, window=200):
    X, Y = [], []
    for i in range(len(series) - window - 1):
        X.append(series[i:i+window, 0])
        Y.append(series[i+window, 0])  # 1-step target
    return np.array(X), np.array(Y)

window = 200

X_train, y_train = make_dataset_1step(train_scaled, window)
X_val,   y_val   = make_dataset_1step(val_scaled, window)
X_test,  y_test  = make_dataset_1step(test_scaled, window)

X_train = X_train[..., None]
X_val   = X_val[..., None]
X_test  = X_test[..., None]


# ============================================================
# 5. Loaders
# ============================================================
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(StockDataset(X_val,   y_val), batch_size=64, shuffle=False)
test_loader  = DataLoader(StockDataset(X_test,  y_test), batch_size=64, shuffle=False)


# ============================================================
# 6. LSTM model (1-step)
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ============================================================
# 7. Training
# ============================================================
def train_epoch(model, loader):
    model.train()
    total = 0
    for X, y in loader:
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)

def eval_epoch(model, loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for X, y in loader:
            preds = model(X)
            loss = criterion(preds, y)
            total += loss.item() * len(X)
    return total / len(loader.dataset)

best_val = float("inf")
for epoch in range(40):
    tr = train_epoch(model, train_loader)
    va = eval_epoch(model, val_loader)
    print(f"{epoch+1:02d} | train {tr:.6f} | val {va:.6f}")
    if va < best_val:
        best_val = va
        torch.save(model.state_dict(), "best_lstm_1step.pt")

model.load_state_dict(torch.load("best_lstm_1step.pt"))
model.eval()


# ============================================================
# 8. Recursive Forecasting (30-step ahead)
# ============================================================
def recursive_predict_30(seq_scaled):
    """
    seq_scaled: shape (window,) scaled
    return: 30-step predicted (scaled) array
    """
    seq = seq_scaled.copy()
    preds = []
    for _ in range(30):
        X = torch.tensor(seq.reshape(1, window, 1), dtype=torch.float32)
        with torch.no_grad():
            next_val = model(X).item()
        preds.append(next_val)

        # shift window
        seq = np.append(seq[1:], next_val)
    return np.array(preds)


# ============================================================
# 9. Test set 30-step prediction & RMSE
# ============================================================
# True values: build 30-step ground truth
def build_test_truth(series_scaled, window=200, horizon=30):
    Y = []
    for i in range(len(series_scaled) - window - horizon):
        Y.append(series_scaled[i+window : i+window+horizon, 0])
    return np.array(Y)

y_test_30 = build_test_truth(test_scaled, window, 30)  # scaled ground truth

pred_list = []
naive_list = []

for i in range(len(y_test_30)):
    seq = test_scaled[i:i+window, 0]
    pred_scaled = recursive_predict_30(seq)
    pred_list.append(pred_scaled)

    naive_last = seq[-1]
    naive_list.append(np.repeat(naive_last, 30))

pred_scaled_all = np.array(pred_list)
naive_scaled_all = np.array(naive_list)

# inverse transform
def inv(flat):
    return scaler.inverse_transform(flat.reshape(-1,1))[:,0]

true_flat  = inv(y_test_30.reshape(-1))
pred_flat  = inv(pred_scaled_all.reshape(-1))
naive_flat = inv(naive_scaled_all.reshape(-1))


# RMSE
rmse = lambda a,b: np.sqrt(mean_squared_error(a,b))

print("\n[TEST RMSE]")
print("Naive:", rmse(true_flat, naive_flat))
print("LSTM :", rmse(true_flat, pred_flat))


# ============================================================
# 10. Monthly rolling: recursive 30-day ahead prediction
# ============================================================
all_scaled = scaler.transform(df)
prices_real = df["price"].values

start_for_months = pd.Timestamp("2019-01-01")
month_ends = pd.date_range(start_for_months, dates[-1], freq="M")

month_dates = []
real_month = []
pred_month = []
naive_month = []

for m_end in month_ends:
    if m_end < dates[0] or m_end > dates[-1]:
        continue

    idx = dates.searchsorted(m_end, side="right") - 1
    if idx < 0 or idx >= n_total:
        continue

    start_i = idx + 1 - window
    future_start = idx + 1
    target_idx = future_start + 30 - 1

    if start_i < 0 or target_idx >= n_total:
        continue

    seq = all_scaled[start_i:start_i+window, 0]
    pred_scaled = recursive_predict_30(seq)
    pred_real = inv(pred_scaled)

    pred_val = pred_real[-1]
    real_val = prices_real[target_idx]
    naive_val = prices_real[idx]

    month_dates.append(dates[target_idx])
    real_month.append(real_val)
    pred_month.append(pred_val)
    naive_month.append(naive_val)

real_month  = np.array(real_month)
pred_month  = np.array(pred_month)
naive_month = np.array(naive_month)

print("\n[Monthly 30-day RMSE]")
print("Naive:", rmse(real_month, naive_month))
print("LSTM :", rmse(real_month, pred_month))


# ============================================================
# 11. Monthly plot (English labels)
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(month_dates, real_month,  label="Real (30d)")
plt.plot(month_dates, pred_month,  label="LSTM (30d ahead)")
plt.plot(month_dates, naive_month, label="Naive (prev month)")
plt.xticks(rotation=45)
plt.ylabel("Index Level")
plt.xlabel("Target Date (30 days ahead)")
plt.title("Monthly Rolling 30-Day Ahead Forecast (Recursive)")
plt.legend()
plt.tight_layout()
plt.show()
