import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# ============================================================
# 1. Load Samsung Electronics OHLC data
# ============================================================
# proj_data.csv columns (assumed): Date, Open, High, Low, Close, Volume, Change
df_raw = pd.read_csv("proj_data.csv")

df_raw["Date"] = pd.to_datetime(df_raw["Date"])
df_raw = df_raw.sort_values("Date").set_index("Date")

# Use OHLC only
df = df_raw[["Open", "High", "Low", "Close"]].dropna()
dates = df.index
n_total = len(df)
print("Total rows:", n_total)
print(df.head())


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
train_scaled = scaler.fit_transform(train)  # (N_train, 4)
val_scaled   = scaler.transform(val)       # (N_val, 4)
test_scaled  = scaler.transform(test)      # (N_test, 4)

# Close column index in scaled arrays
CLOSE_COL = list(df.columns).index("Close")  # should be 3


# ============================================================
# 4. Multi-step dataset: window -> next 30 Close values
# ============================================================
def make_dataset_multistep(series, window=252, horizon=30, target_col=3):
    """
    series: numpy array (N, num_features)
    X: (samples, window, num_features)
    y: (samples, horizon) for target_col (scaled)
    """
    X_list, Y_list = [], []
    for i in range(len(series) - window - horizon + 1):
        X_list.append(series[i:i+window, :])
        Y_list.append(series[i+window : i+window+horizon, target_col])
    return np.array(X_list), np.array(Y_list)

window = 252  # about 1 trading year
horizon = 30  # 30-day ahead path

X_train, y_train = make_dataset_multistep(train_scaled, window, horizon, CLOSE_COL)
X_val,   y_val   = make_dataset_multistep(val_scaled,   window, horizon, CLOSE_COL)
X_test,  y_test  = make_dataset_multistep(test_scaled,  window, horizon, CLOSE_COL)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape,   "y_val  :", y_val.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)


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
# 6. LSTM model (multi-step output)
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, horizon=30):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]       # last timestep
        out = self.fc(out)        # (batch, horizon)
        return out                # scaled Close predictions


model = LSTMModel(input_dim=4, hidden_dim=64, num_layers=2, horizon=horizon)
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
num_epochs = 40

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss   = eval_epoch(model,   val_loader)
    print(f"{epoch+1:02d} | train {train_loss:.6f} | val {val_loss:.6f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "samsung_ohlc_lstm_multistep.pt")

model.load_state_dict(torch.load("samsung_ohlc_lstm_multistep.pt"))
model.eval()


# ============================================================
# 8. Test prediction (30-day path) + Naive baseline
# ============================================================
y_preds_scaled_list = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        pred = model(X_batch)  # (batch, horizon)
        y_preds_scaled_list.append(pred.numpy())

y_preds_scaled = np.concatenate(y_preds_scaled_list, axis=0)  # (N_test, horizon)

# Naive: last Close of window repeated for 30 days
last_close_scaled = X_test[:, -1, CLOSE_COL]
y_naive_scaled = np.repeat(last_close_scaled[:, None], horizon, axis=1)


# ============================================================
# 9. Inverse transform for Close
# ============================================================
def inverse_close_only(scaled_close_1d):
    """
    scaled_close_1d: (N,) array for Close column (scaled)
    Return: (N,) real Close values via MinMaxScaler
    """
    dummy = np.zeros((len(scaled_close_1d), train_scaled.shape[1]))
    dummy[:, CLOSE_COL] = scaled_close_1d
    inv = scaler.inverse_transform(dummy)[:, CLOSE_COL]
    return inv

N_test = y_test.shape[0]

y_test_real_flat  = inverse_close_only(y_test.reshape(-1))
y_pred_real_flat  = inverse_close_only(y_preds_scaled.reshape(-1))
y_naive_real_flat = inverse_close_only(y_naive_scaled.reshape(-1))

y_test_real  = y_test_real_flat.reshape(N_test, horizon)
y_pred_real  = y_pred_real_flat.reshape(N_test, horizon)
y_naive_real = y_naive_real_flat.reshape(N_test, horizon)


# ============================================================
# 10. Metrics: RMSE + Direction Accuracy
# ============================================================
def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


# --- (1) Overall RMSE for all 30 horizons combined ---
rmse_naive_all = rmse(y_test_real_flat, y_naive_real_flat)
rmse_lstm_all  = rmse(y_test_real_flat, y_pred_real_flat)

print("\n[TEST RMSE (30-step path, all points)]")
print("Naive:", rmse_naive_all)
print("LSTM :", rmse_lstm_all)


# --- (2) 30-day ahead direction accuracy (relative to last known Close) ---
# For each test sample i:
# origin index in df: last day of the window
origin_indices = [val_end + i + window - 1 for i in range(N_test)]
origin_prices  = df["Close"].iloc[origin_indices].to_numpy()

real_30  = y_test_real[:, -1]
pred_30  = y_pred_real[:, -1]
naive_30 = y_naive_real[:, -1]

real_dir  = np.sign(real_30  - origin_prices)
pred_dir  = np.sign(pred_30  - origin_prices)
naive_dir = np.sign(naive_30 - origin_prices)

# treat 0 difference as 0; count it as correct if both are 0
def direction_accuracy(true_dir, pred_dir):
    return np.mean(true_dir == pred_dir)

acc_lstm  = direction_accuracy(real_dir, pred_dir)
acc_naive = direction_accuracy(real_dir, naive_dir)

print("\n[TEST Direction Accuracy (30-day ahead up/down)]")
print("Naive:", acc_naive)
print("LSTM :", acc_lstm)


# ============================================================
# 11. Monthly rolling 30-day ahead metrics
# ============================================================
all_scaled  = scaler.transform(df)                # (n_total, 4)
prices_real = df["Close"].values

start_for_months = pd.Timestamp("2002-01-01")     # ensure enough history
month_ends = pd.date_range(start_for_months, dates[-1], freq="ME")  # month-end

month_dates = []
real_month  = []
pred_month  = []
naive_month = []
dir_real_m  = []
dir_pred_m  = []
dir_naive_m = []

for m_end in month_ends:
    if m_end < dates[0] or m_end > dates[-1]:
        continue

    # index of last trading day on/before m_end
    idx = dates.searchsorted(m_end, side="right") - 1
    if idx < 0 or idx >= n_total:
        continue

    start_i = idx + 1 - window
    future_start = idx + 1
    target_idx   = future_start + horizon - 1   # 30th day ahead

    if start_i < 0 or target_idx >= n_total:
        continue

    seq_scaled = all_scaled[start_i:start_i+window, :]  # (window, 4)
    X_seq = torch.tensor(seq_scaled.reshape(1, window, 4), dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(X_seq).numpy().reshape(-1)   # (horizon,)

    pred_real_30d = inverse_close_only(pred_scaled)
    pred_val  = pred_real_30d[-1]
    real_val  = prices_real[target_idx]
    naive_val = prices_real[idx]

    month_dates.append(dates[target_idx])
    real_month.append(real_val)
    pred_month.append(pred_val)
    naive_month.append(naive_val)

    # directions for month-level
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

rmse_naive_month = rmse(real_month, naive_month)
rmse_lstm_month  = rmse(real_month, pred_month)

acc_naive_month = direction_accuracy(dir_real_m, dir_naive_m)
acc_lstm_month  = direction_accuracy(dir_real_m, dir_pred_m)

print("\n[Monthly 30-day RMSE]")
print("Naive:", rmse_naive_month)
print("LSTM :", rmse_lstm_month)

print("\n[Monthly 30-day Direction Accuracy]")
print("Naive:", acc_naive_month)
print("LSTM :", acc_lstm_month)


# ============================================================
# 12. Plot monthly rolling 30-day ahead forecast
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(month_dates, real_month,  label="Real (30d)",
         linestyle = '-.', marker = 's', markersize = 7)
plt.plot(month_dates, pred_month,  label="LSTM (30d ahead)",
         linestyle = '-', marker = 'o', markersize = 7)
plt.plot(month_dates, naive_month, label="Naive (prev month)",
         linestyle = '--', marker = '^', markersize = 7)
plt.xticks(rotation=45)
plt.ylabel("Price")
plt.xlabel("Target date (30 days ahead)")
plt.title("Samsung Electronics - Monthly Rolling 30-Day Ahead Forecast (OHLC input)")
plt.legend()
plt.tight_layout()
plt.savefig("figure.pdf", dpi=300, bbox_inches="tight")
plt.show()
