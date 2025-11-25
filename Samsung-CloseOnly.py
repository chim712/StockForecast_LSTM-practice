import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ============================================================
# 1. Load Samsung Electronics data from CSV
# ============================================================
# CSV columns: Date, Open, High, Low, Close, Volume, Change
df_raw = pd.read_csv("proj_data.csv")

# parse Date and set index
df_raw["Date"] = pd.to_datetime(df_raw["Date"])
df_raw = df_raw.sort_values("Date").set_index("Date")

# we use only Close price
df = df_raw[["Close"]].rename(columns={"Close": "price"})
df = df.dropna()

dates = df.index
n_total = len(df)
print("Total rows:", n_total)
print(df.head())

# ============================================================
# 2. Train / Val / Test split (time-series)
# ============================================================
n = len(df)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]

print("Train:", train.shape, "Val:", val.shape, "Test:", test.shape)

# ============================================================
# 3. Scaling (fit on train)
# ============================================================
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
val_scaled   = scaler.transform(val)
test_scaled  = scaler.transform(test)

# ============================================================
# 4. 1-step dataset (window -> next day)
# ============================================================
def make_dataset_1step(series, window=252):
    X, Y = [], []
    for i in range(len(series) - window - 1):
        X.append(series[i:i+window, 0])
        Y.append(series[i+window, 0])
    return np.array(X), np.array(Y)

window = 252  # about 1 trading year

X_train, y_train = make_dataset_1step(train_scaled, window)
X_val,   y_val   = make_dataset_1step(val_scaled, window)
X_test,  y_test  = make_dataset_1step(test_scaled, window)

X_train = X_train[..., None]
X_val   = X_val[..., None]
X_test  = X_test[..., None]

print("X_train:", X_train.shape, "y_train:", y_train.shape)

# ============================================================
# 5. Datasets / Loaders
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
# 7. Training loop
# ============================================================
def train_epoch(model, loader):
    model.train()
    total = 0.0
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
    total = 0.0
    with torch.no_grad():
        for X, y in loader:
            preds = model(X)
            loss = criterion(preds, y)
            total += loss.item() * len(X)
    return total / len(loader.dataset)

best_val = float("inf")
for epoch in range(40):
    train_loss = train_epoch(model, train_loader)
    val_loss   = eval_epoch(model,   val_loader)
    print(f"{epoch+1:02d} | train {train_loss:.6f} | val {val_loss:.6f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "samsung_lstm_1step.pt")

model.load_state_dict(torch.load("samsung_lstm_1step.pt"))
model.eval()

# ============================================================
# 8. Recursive 30-day forecasting on test set
# ============================================================
def recursive_predict_30(seq_scaled, window=252):
    seq = seq_scaled.copy()
    preds = []
    for _ in range(30):
        X = torch.tensor(seq.reshape(1, window, 1), dtype=torch.float32)
        with torch.no_grad():
            next_val = model(X).item()
        preds.append(next_val)
        seq = np.append(seq[1:], next_val)
    return np.array(preds)

def build_test_truth(series_scaled, window=252, horizon=30):
    Y = []
    for i in range(len(series_scaled) - window - horizon):
        Y.append(series_scaled[i+window : i+window+horizon, 0])
    return np.array(Y)

horizon = 30
y_test_30 = build_test_truth(test_scaled, window, horizon)

pred_list = []
naive_list = []
for i in range(len(y_test_30)):
    seq = test_scaled[i:i+window, 0]
    pred_scaled = recursive_predict_30(seq, window)
    pred_list.append(pred_scaled)
    naive_last = seq[-1]
    naive_list.append(np.repeat(naive_last, horizon))

pred_scaled_all  = np.array(pred_list)   # (N_test, 30)
naive_scaled_all = np.array(naive_list)  # (N_test, 30)

def inv(flat):
    return scaler.inverse_transform(flat.reshape(-1,1))[:,0]

true_flat  = inv(y_test_30.reshape(-1))
pred_flat  = inv(pred_scaled_all.reshape(-1))
naive_flat = inv(naive_scaled_all.reshape(-1))

N_test = y_test_30.shape[0]
y_test_real  = true_flat.reshape(N_test, horizon)
y_pred_real  = pred_flat.reshape(N_test, horizon)
y_naive_real = naive_flat.reshape(N_test, horizon)

rmse = lambda a,b: np.sqrt(mean_squared_error(a,b))

print("\n[TEST RMSE (Samsung, 30-step path)]")
print("Naive:", rmse(true_flat, naive_flat))
print("LSTM :", rmse(true_flat, pred_flat))

# ---------- Direction accuracy (30-day ahead) ----------
dates_all   = df.index
prices_real = df["price"].values

origin_indices = [val_end + i + window - 1 for i in range(N_test)]
origin_prices  = prices_real[origin_indices]

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

print("\n[TEST Direction Accuracy (30-day ahead up/down)]")
print("Naive:", acc_naive)
print("LSTM :", acc_lstm)

# ============================================================
# 9. Monthly rolling 30-day ahead forecast (recursive)
# ============================================================
all_scaled   = scaler.transform(df)
prices_real  = df["price"].values
dates_all    = df.index

start_for_months = pd.Timestamp("2002-01-01")  # enough history before this
month_ends = pd.date_range(start_for_months, dates_all[-1], freq="ME")  # month-end

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

    start_i = idx + 1 - window
    future_start = idx + 1
    target_idx = future_start + horizon - 1

    if start_i < 0 or target_idx >= n_total:
        continue

    seq = all_scaled[start_i:start_i+window, 0]
    pred_scaled = recursive_predict_30(seq, window)
    pred_real   = inv(pred_scaled)

    pred_val  = pred_real[-1]
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

print("\n[Monthly 30-day RMSE (Samsung)]")
print("Naive:", rmse(real_month, naive_month))
print("LSTM :", rmse(real_month, pred_month))

acc_naive_m = direction_accuracy(dir_real_m, dir_naive_m)
acc_lstm_m  = direction_accuracy(dir_real_m, dir_pred_m)

print("\n[Monthly 30-day Direction Accuracy (Samsung)]")
print("Naive:", acc_naive_m)
print("LSTM :", acc_lstm_m)

# ============================================================
# 10. Plot monthly rolling 30-day ahead forecast
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
plt.title("Samsung Electronics - Monthly Rolling 30-Day Ahead Forecast (Close only)")
plt.legend()
plt.tight_layout()
plt.savefig("figure.pdf", dpi=300, bbox_inches="tight")
plt.show()
