import yfinance as yf
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

startDate = "2003-01-01"

# ============================================================
# 1. Load OHLC data (005930.KS)
# ============================================================
ticker = "005930.KS"
data = yf.download(ticker, start=startDate, end="2024-03-30")

df = data[["Open", "High", "Low", "Close"]].dropna()
dates = df.index
n_total = len(df)

print("Total rows:", n_total)
print(df.head())

CLOSE_COL = list(df.columns).index(("Close", ticker))  # 3

# ============================================================
# 2. Train / Val / Test split
# ============================================================
n = len(df)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]

print("Train:", train.shape, "Val:", val.shape, "Test:", test.shape)

# ============================================================
# 3. Scaling (fit only on train)
# ============================================================
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)  # (N, 4)
val_scaled   = scaler.transform(val)
test_scaled  = scaler.transform(test)

# ============================================================
# 4. Dataset: 1-step ahead OHLC forecasting
#    X: window days of OHLC
#    Y: next day OHLC (4-dim)
# ============================================================
def make_dataset_1step_ohlc(series, window=252):
    X_list, Y_list = [], []
    for i in range(len(series) - window - 1):
        X_list.append(series[i:i+window, :])      # (window, 4)
        Y_list.append(series[i+window, :])        # (4,)
    return np.array(X_list), np.array(Y_list)

window = 252

X_train, y_train = make_dataset_1step_ohlc(train_scaled, window)
X_val,   y_val   = make_dataset_1step_ohlc(val_scaled,   window)
X_test,  y_test  = make_dataset_1step_ohlc(test_scaled,  window)

print("X_train:", X_train.shape, "y_train:", y_train.shape)

# ============================================================
# 5. Dataset / DataLoader
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
val_loader   = DataLoader(StockDataset(X_val,   y_val),   batch_size=64, shuffle=False)
test_loader  = DataLoader(StockDataset(X_test,  y_test),  batch_size=64, shuffle=False)

# ============================================================
# 6. LSTM model (1-step OHLC)
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, output_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # last time step
        out = self.fc(out)       # (batch, 4)
        return out

model = LSTMModel(input_dim=4, hidden_dim=64, num_layers=2, output_dim=4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================
# 7. Training loop
# ============================================================
def train_epoch(model, loader):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)

best_val = float("inf")
for epoch in range(40):
    tr = train_epoch(model, train_loader)
    va = eval_epoch(model,   val_loader)
    print(f"{epoch+1:02d} | train {tr:.6f} | val {va:.6f}")
    if va < best_val:
        best_val = va
        torch.save(model.state_dict(), "best_lstm_1step_ohlc.pt")

model.load_state_dict(torch.load("best_lstm_1step_ohlc.pt"))
model.eval()

# ============================================================
# 8. Recursive 30-step forecasting on test set
#    - truth: 실제 30일 OHLC 경로 (scaled)
#    - prediction: 모델을 재귀적으로 30번 굴린 경로 (scaled)
#    - naive: 마지막 하루 OHLC를 30번 반복한 경로 (scaled)
# ============================================================
horizon = 30

def recursive_predict_30_ohlc(seq_scaled, model, window=252, horizon=30):
    """
    seq_scaled: (window, 4) - 최근 window일의 스케일된 OHLC
    반환: (horizon, 4) - 스케일된 예측 OHLC 경로
    """
    seq = seq_scaled.copy()
    preds = []
    for _ in range(horizon):
        X = torch.tensor(seq.reshape(1, window, 4), dtype=torch.float32)
        with torch.no_grad():
            next_ohlc = model(X).numpy().reshape(4)
        preds.append(next_ohlc)
        seq = np.vstack([seq[1:], next_ohlc])  # 마지막 예측을 시퀀스에 추가
    return np.array(preds)  # (horizon, 4)

def build_test_truth_ohlc(series_scaled, window=252, horizon=30):
    """
    series_scaled: test_scaled (len_T, 4)
    반환: (N, horizon, 4)
    N = len(series_scaled) - window - horizon
    """
    Y_list = []
    for i in range(len(series_scaled) - window - horizon):
        Y_list.append(series_scaled[i+window : i+window+horizon, :])
    return np.array(Y_list)

# build ground-truth multi-step OHLC from test split
y_test_30_scaled = build_test_truth_ohlc(test_scaled, window=window, horizon=horizon)
N_test = y_test_30_scaled.shape[0]
print("y_test_30_scaled:", y_test_30_scaled.shape)

pred_list = []
naive_list = []

for i in range(N_test):
    seq = test_scaled[i:i+window, :]        # (window, 4)
    pred_scaled = recursive_predict_30_ohlc(seq, model, window=window, horizon=horizon)
    pred_list.append(pred_scaled)

    # naive: 마지막 하루 OHLC를 30번 반복
    last_row = seq[-1, :]  # (4,)
    naive_path = np.repeat(last_row.reshape(1,4), horizon, axis=0)
    naive_list.append(naive_path)

pred_scaled_all  = np.array(pred_list)   # (N_test, horizon, 4)
naive_scaled_all = np.array(naive_list)  # (N_test, horizon, 4)

# ============================================================
# 9. Inverse transform & evaluation (Close 기준)
# ============================================================
def inv_ohlc_2d(arr2d):
    """
    arr2d: (K, 4) 스케일된 OHLC
    반환: (K, 4) 원래 스케일
    """
    return scaler.inverse_transform(arr2d)

# (N_test, horizon, 4) -> (N_test*horizon, 4) -> inverse -> 다시 reshape
true_flat2d  = y_test_30_scaled.reshape(-1, 4)
pred_flat2d  = pred_scaled_all.reshape(-1, 4)
naive_flat2d = naive_scaled_all.reshape(-1, 4)

true_real2d  = inv_ohlc_2d(true_flat2d)
pred_real2d  = inv_ohlc_2d(pred_flat2d)
naive_real2d = inv_ohlc_2d(naive_flat2d)

y_test_real_ohlc  = true_real2d.reshape(N_test, horizon, 4)
y_pred_real_ohlc  = pred_real2d.reshape(N_test, horizon, 4)
y_naive_real_ohlc = naive_real2d.reshape(N_test, horizon, 4)

# Close만 뽑아서 RMSE 계산
true_close_flat  = true_real2d[:, CLOSE_COL]
pred_close_flat  = pred_real2d[:, CLOSE_COL]
naive_close_flat = naive_real2d[:, CLOSE_COL]

y_test_real  = true_close_flat.reshape(N_test, horizon)   # (N_test, horizon)
y_pred_real  = pred_close_flat.reshape(N_test, horizon)
y_naive_real = naive_close_flat.reshape(N_test, horizon)

rmse = lambda a,b: np.sqrt(mean_squared_error(a,b))

print(f"\n[TEST RMSE ({ticker}, 30-step path, Close only)]")
print("Naive:", rmse(true_close_flat, naive_close_flat))
print("LSTM :", rmse(true_close_flat, pred_close_flat))

# ---------- Direction accuracy (30-day ahead, Close) ----------
prices_real_all = df["Close"].values

origin_indices = [val_end + i + window - 1 for i in range(N_test)]
origin_prices  = prices_real_all[origin_indices]

real_30  = y_test_real[:, -1]
pred_30  = y_pred_real[:, -1]
naive_30 = y_naive_real[:, -1]

real_dir  = np.sign(real_30  - origin_prices)
pred_dir  = np.sign(pred_30  - origin_prices)
naive_dir = np.sign(naive_30 - origin_prices)

def direction_accuracy(true_dir, pred_dir):
    return np.mean(true_dir == pred_dir)

acc_naive = direction_accuracy(real_dir, naive_dir)
acc_lstm  = direction_accuracy(real_dir, pred_dir)

print(f"\n[TEST Direction Accuracy (30-day ahead up/down, Close, {ticker})]")
print("Naive:", acc_naive)
print("LSTM :", acc_lstm)

# ============================================================
# 10. Monthly rolling: recursive 30-day ahead prediction (Close 기준)
# ============================================================
all_scaled  = scaler.transform(df)            # 전체 구간 (train+val+test)
prices_real = df["Close"].values
dates_all   = df.index

start_for_months = pd.Timestamp(startDate)
month_ends = pd.date_range(start_for_months, dates_all[-1], freq="ME")

month_dates = []
real_month  = []
pred_month  = []
naive_month = []
dir_real_m  = []
dir_pred_m  = []
dir_naive_m = []

for m_end in month_ends:
    if m_end < dates_all[0] or m_end > dates_all[-1]:
        continue

    idx = dates_all.searchsorted(m_end, side="right") - 1
    if idx < 0 or idx >= n_total:
        continue

    start_i      = idx + 1 - window           # 과거 window 시작 index
    future_start = idx + 1                    # 예측 시작일 index
    target_idx   = future_start + horizon - 1

    if start_i < 0 or target_idx >= n_total:
        continue

    seq_scaled = all_scaled[start_i:start_i+window, :]  # (window, 4)
    pred_scaled_path = recursive_predict_30_ohlc(seq_scaled, model, window=window, horizon=horizon)
    pred_real_path   = inv_ohlc_2d(pred_scaled_path)    # (horizon, 4)

    pred_val  = pred_real_path[-1, CLOSE_COL]
    real_val  = prices_real[target_idx]
    naive_val = prices_real[idx]

    month_dates.append(dates_all[target_idx])
    real_month.append(real_val)
    pred_month.append(pred_val)
    naive_month.append(naive_val)

    base_price = prices_real[idx]
    dir_real_m.append(np.sign(real_val - base_price))
    dir_pred_m.append(np.sign(pred_val - base_price))
    dir_naive_m.append(np.sign(naive_val - base_price))

real_month  = np.array(real_month)
pred_month  = np.array(pred_month)
naive_month = np.array(naive_month)
dir_real_m  = np.array(dir_real_m)
dir_pred_m  = np.array(dir_pred_m)
dir_naive_m = np.array(dir_naive_m)

print(f"\n[Monthly 30-day RMSE ({ticker}, Close only)]")
print("Naive:", rmse(real_month, naive_month))
print("LSTM :", rmse(real_month, pred_month))

acc_naive_m = direction_accuracy(dir_real_m, dir_naive_m)
acc_lstm_m  = direction_accuracy(dir_real_m, dir_pred_m)

print(f"\n[Monthly 30-day Direction Accuracy (Close, {ticker})]")
print("Naive:", acc_naive_m)
print("LSTM :", acc_lstm_m)

# ============================================================
# 10-1. Save result (Close-only 결과 저장)
# ============================================================
month_dates_arr = np.array([d.strftime("%Y-%m-%d") for d in month_dates])

np.savez(
    "test_OHLC_1step.npz",
    y_test_real=y_test_real,       # (N_test, 30) Close
    y_pred_real=y_pred_real,
    y_naive_real=y_naive_real,
    month_dates=month_dates_arr,
    real_month=real_month,
    pred_month=pred_month,
    naive_month=naive_month,
)

print("\nSaved to test_OHLC_1step.npz")

# ============================================================
# 11. Monthly plot (Close 기준)
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(month_dates, real_month,  label="Real (30d)")
plt.plot(month_dates, pred_month,  label="LSTM (30d ahead)")
plt.plot(month_dates, naive_month, label="Naive (prev month)")
plt.xticks(rotation=45)
plt.ylabel("Index Level (Close)")
plt.xlabel("Target Date (30 days ahead)")
plt.title("005930 - Monthly Rolling 30-Day Ahead Forecast (1-step OHLC, Close)")
plt.legend()
plt.tight_layout()
plt.show()
