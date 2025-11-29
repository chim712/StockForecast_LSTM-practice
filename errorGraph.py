import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # ============================================================
    # 1. npz 파일 불러오기
    # ============================================================
    data = np.load("Samsung_OHLC.npz", allow_pickle=True)

    # month_dates: 문자열 배열 (예: "2005-01-31")
    month_dates_arr = data["month_dates"]

    # ★ 여기서 1차원으로 강제 변환 (reshape / ravel)
    real_month  = np.ravel(data["real_month"])
    pred_month  = np.ravel(data["pred_month"])
    naive_month = np.ravel(data["naive_month"])

    # 길이 확인 (디버깅용, 필요 없으면 주석 처리)
    # print(len(month_dates_arr), len(real_month), len(pred_month), len(naive_month))

    # 문자열 → datetime
    month_dates = pd.to_datetime(month_dates_arr)

    # ============================================================
    # 2. 월별 예측 오차 계산 (실제 - 예측)
    # ============================================================
    err_lstm  = real_month - pred_month
    err_naive = real_month - naive_month

    # ============================================================
    # 3. DataFrame으로 정리 후 상반기/하반기 단위 그룹핑
    # ============================================================
    df_err = pd.DataFrame({
        "date": month_dates,
        "err_lstm": err_lstm,
        "err_naive": err_naive,
    })

    # 연도, 반기(H1/H2) 컬럼 추가
    df_err["year"] = df_err["date"].dt.year
    df_err["half"] = np.where(df_err["date"].dt.month <= 6, "H1", "H2")

    # year, half 기준으로 평균 오차 계산
    grouped = (
        df_err
        .groupby(["year", "half"], as_index=False)
        .agg({
            "err_lstm": "mean",
            "err_naive": "mean",
        })
        .sort_values(["year", "half"])
    )

    # X축 레이블: 예) 2005-H1, 2005-H2, ...
    labels = grouped["year"].astype(str) + "-" + grouped["half"]

    # ============================================================
    # 4. 상반기/하반기 단위 막대 그래프 (Y=0 기준 양/음 오차)
    # ============================================================
    x = np.arange(len(grouped))
    width = 0.35

    plt.figure(figsize=(10, 5))

    # LSTM 반기 평균 오차
    plt.bar(
        x - width / 2,
        grouped["err_lstm"],
        width=width,
        label="LSTM Error (Real - Pred)",
        color="0.7"
    )

    # Naive 반기 평균 오차
    plt.bar(
        x + width / 2,
        grouped["err_naive"],
        width=width,
        label="Naive Error (Real - Prev Month)",
        color="0.1"
    )

    plt.xticks(x, labels, rotation=45)
    plt.axhline(0, color="black", linewidth=1)

    plt.ylabel("Forecast Error (Real - Pred)")
    plt.xlabel("Half-Year (H1: Jan–Jun, H2: Jul–Dec)")
    plt.title("Samsung Electronics – 30-Day Ahead Forecast Error (Half-Year Aggregated)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("samsung_ohlc_halfyear_error_barchart.pdf",
                dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
