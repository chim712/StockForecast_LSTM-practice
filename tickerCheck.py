import yfinance as yf


ticker = "005935.KS"
data = yf.download(ticker, start="2003-01-01", end="2024-04-30")

df_raw = data[["Open", "High", "Low", "Close"]].dropna()
df = df_raw.copy()
dates = df.index
n_total = len(df)

print("Total rows:", n_total)
print(df)
df.to_csv(f"{ticker}.csv")