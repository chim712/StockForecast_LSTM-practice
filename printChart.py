import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

markInterval = 96
markSize = 4
lineWidth = 1
Title = "SK Hynix OHLC Input"

data = np.load("Hynix_OHLC(03-24).npz", allow_pickle=True)

month_dates = pd.to_datetime(data["month_dates"])
real_month  = data["real_month"]
pred_month  = data["pred_month"]
naive_month = data["naive_month"]

plt.figure(figsize=(10,6))
# plt.plot(month_dates, real_month,  label="Real (30d)",
#          linestyle='-.', linewidth = lineWidth , marker='s', markevery=(0, markInterval), markersize=markSize)
# plt.plot(month_dates, pred_month,  label="LSTM (30d ahead)",
#          linestyle='-', linewidth = lineWidth , marker='o', markevery=(int(markInterval/3), markInterval), markersize=markSize)
# plt.plot(month_dates, naive_month, label="Naive (prev month)",
#          linestyle='--', linewidth = lineWidth , marker='^', markevery=(int(markInterval/3*2), markInterval), markersize=markSize)

plt.plot(month_dates, real_month,  label="Real (30d)",
         linestyle='-', linewidth = lineWidth, color='black')
plt.plot(month_dates, pred_month,  label="LSTM (30d ahead)",
         linestyle='--', linewidth = lineWidth, color='gray')
plt.plot(month_dates, naive_month, label="Naive (prev month)",
         linestyle=':', linewidth = lineWidth, color='black')


plt.grid(True, linestyle=':', linewidth = 0.5, which='major')
plt.xticks(rotation=45)
plt.ylabel("Index Level")
plt.xlabel("Target date (30 days ahead)")
plt.title(Title)
plt.legend()
plt.tight_layout()
plt.savefig("figure.pdf", dpi=300, bbox_inches="tight")
plt.show()
