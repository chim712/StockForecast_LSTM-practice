import yfinance as yf
import pandas as pd
import numpy as np


# 1. 데이터 수집 및 전처리 =============================
# SPX: S&P 500
ticker = "^SPX"  # 또는 "^KS11" 같은 지수
data = yf.download(ticker, start="2018-01-01", end="2024-01-01")
print(data.head())

# 종가만 사용함
df = data[["Close"]].rename(columns={"Close": "price"})
df = df.dropna()



# 2. 학습, 검증, 테스트 시계열 분할 =======================
n = len(df)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]




# 3. 스케일링 =================================

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
val_scaled   = scaler.transform(val)
test_scaled  = scaler.transform(test)


# 4. 시퀀스(Window) 생성 =======================
import numpy as np

def make_dataset(series, window_size=30):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size, 0])  # 30일
        y.append(series[i+window_size, 0])    # 그 다음날
    return np.array(X), np.array(y)

window = 30
X_train, y_train = make_dataset(train_scaled, window)
X_val,   y_val   = make_dataset(val_scaled, window)
X_test,  y_test  = make_dataset(test_scaled, window)

# LSTM 입력 형태 맞추기 (samples, timesteps, features)
X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
X_test  = X_test[..., np.newaxis]





#5-1 Naive / MA5 Baseline ========================
# 테스트 구간 naive 예측
y_test_naive = X_test[:, -1, 0]  # 마지막 시점 값
y_test_ma5 = X_test[:, -5:, 0].mean(axis=1)

import torch
from torch.utils.data import Dataset, DataLoader

# Pytorch Dataset / DataLoader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = StockDataset(X_train, y_train)
val_ds   = StockDataset(X_val, y_val)
test_ds  = StockDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)


#5-2 LSTM 모델 정의
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)         # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]           # 마지막 시점
        out = self.fc(out)            # (batch, 1)
        return out.squeeze(-1)

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 6. 학습 루프와 Early Stopping ==============================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)

best_val = float("inf")
for epoch in range(50):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = eval_epoch(model, val_loader, criterion)
    print(f"{epoch+1:02d} | train {train_loss:.6f} | val {val_loss:.6f}")
    # 간단한 early stopping
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_lstm.pt")

#7. 테스트 구간 예측, 역정규화, 평가 ======================================
model.load_state_dict(torch.load("best_lstm.pt"))
model.eval()

y_preds_scaled = []
with torch.no_grad():
    for X, _ in test_loader:
        pred = model(X)
        y_preds_scaled.append(pred.numpy())
import numpy as np
y_preds_scaled = np.concatenate(y_preds_scaled, axis=0)

# 역정규화
# scaler.inverse_transform은 2D 입력을 기대하므로 reshape 필요
y_test_scaled_2d  = y_test.reshape(-1, 1)
y_preds_scaled_2d = y_preds_scaled.reshape(-1, 1)
y_test_real  = scaler.inverse_transform(y_test_scaled_2d)[:,0]
y_pred_real  = scaler.inverse_transform(y_preds_scaled_2d)[:,0]
y_naive_real = scaler.inverse_transform(y_test_naive.reshape(-1,1))[:,0]
y_ma5_real   = scaler.inverse_transform(y_test_ma5.reshape(-1,1))[:,0]

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

print("Naive RMSE:", rmse(y_test_real, y_naive_real))
print("MA5   RMSE:", rmse(y_test_real, y_ma5_real))
print("LSTM  RMSE:", rmse(y_test_real, y_pred_real))


# 8. 결과 시각화 ======================================
import matplotlib.pyplot as plt

plt.plot(y_test_real[:200], label="real")
plt.plot(y_naive_real[:200], label="naive")
plt.plot(y_pred_real[:200], label="lstm")
plt.legend()
plt.show()
