import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def daily2weekly(sp500_path, rfr_path, vix_path, gspc_path):
    out_dir = os.path.join(os.path.dirname(__file__), "data/")

    # loading data
    sp500_df = pd.read_csv(sp500_path)
    sp500_df.Date = pd.to_datetime(sp500_df.Date)
    sp500_df.set_index("Date", inplace=True)

    rfr_df = pd.read_csv(rfr_path, index_col = 0)
    rfr_df.index = pd.to_datetime(rfr_df.index)

    vix_df = pd.read_csv(vix_path, index_col = 0)
    vix_df.index = pd.to_datetime(vix_df.index)

    gspc_df = pd.read_csv(gspc_path, index_col = 0)
    gspc_df.index = pd.to_datetime(gspc_df.index)

    # checking time range
    if not ((sp500_df.index == rfr_df.index).all() and (rfr_df.index == vix_df.index).all()):
        raise IndexError("All three data should have the same time range.")

    # S&P 500
    sp500_df = sp500_df.resample('w-fri').last()
    nan_cols = sp500_df.columns[sp500_df.isna().any()].tolist()
    candidate_tickers = sorted(list(set(sp500_df.columns.tolist()).difference(set(nan_cols))))
    sp500_df = sp500_df[candidate_tickers]
    sp500_df.iloc[1:, :].to_csv(f"{os.path.join(out_dir, 'sp500_weekly_close.csv')}", encoding = "utf-8")

    sp500_df = (sp500_df - sp500_df.shift(1))/(sp500_df.shift(1))
    sp500_df.iloc[1:, :].to_csv(f"{os.path.join(out_dir, 'sp500_weekly_return_rate.csv')}", encoding = "utf-8")
    print(f"output {os.path.join(out_dir, 'sp500_weekly_return_rate.csv')}")

    # GSPC
    gspc_df = gspc_df.resample('w-fri').last()
    gspc_df.iloc[1:, :].to_csv(f"{os.path.join(out_dir, 'gspc_weekly_close.csv')}", encoding = "utf-8")

    gspc_df = (gspc_df - gspc_df.shift(1))/(gspc_df.shift(1))
    gspc_df.iloc[1:, :].to_csv(f"{os.path.join(out_dir, 'gspc_weekly_return_rate.csv')}", encoding = "utf-8")
    print(f"output {os.path.join(out_dir, 'gspc_weekly_return_rate.csv')}")

    # risk free rate
    # trasforms to deci
    rfr_df = rfr_df / 100
    # anualized rate → daily rate
    rfr_df_daily = (1 + rfr_df) ** (1/252) - 1
    # daily rate → weekly（以週五為基準，取該週所有日報酬的複利）
    rfr_df_weekly = (1 + rfr_df_daily).resample("W-FRI").prod() - 1
    rfr_df_weekly.iloc[1:, :].to_csv(f"{os.path.join(out_dir, 'risk_free_weekly_rate.csv')}", encoding = "utf-8")
    print(f"output {os.path.join(out_dir, 'risk_free_weekly_rate.csv')}")

    # VIX
    vix_df = vix_df.resample('w-fri').last()
    vix_df.iloc[1:, :].to_csv(f"{os.path.join(out_dir, 'vix_weekly_index.csv')}", encoding = "utf-8")
    print(f"output {os.path.join(out_dir, 'vix_weekly_index.csv')}")

if __name__ == "__main__":
    sp500_path = "./data_collection/sp500_20170101-20250901_close.csv"
    rfr_path = "./data_collection/risk_free_rate_20170101-20250901.csv"
    vix_path = "./data_collection/VIX_20170101-20250901.csv"
    gspc_path = "./data_collection/GSPC_20170101-20250901.csv"

    daily2weekly(sp500_path, rfr_path, vix_path, gspc_path)