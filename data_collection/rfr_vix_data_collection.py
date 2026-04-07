import yfinance as yf
import numpy as np
import pandas as pd

# ^IRX = 13-Week (3 個月期) Treasury Bill Yield
# ^FVX = 5-Year Treasury Yield
# ^TNX = 10-Year Treasury Yield
# ^TYX = 30-Year Treasury Yield

output_dir = "data_collection"
rfr_tenor = ["^IRX", "^FVX", "^TNX", "^TYX"]
START = "2017-01-01"
END = "2025-09-01"

lst = []
for t in range(len(rfr_tenor)):
    rf = yf.download(rfr_tenor[t], start=START, end=END)
    if t == 0:
        idx = rf.index.to_numpy().astype("datetime64[D]")
    rf = rf[["Close"]].to_numpy().reshape(-1)
    lst.append(rf)

pd.DataFrame(np.array(lst).T, columns=rfr_tenor, index=idx).to_csv(f"./{output_dir}/risk_free_rate_{START.replace("-", "")}-{END.replace("-", "")}.csv", encoding='utf-8')

# VIX 
vix_df =  yf.download("^VIX", start=START, end=END)
vix_arr =vix_df[["Close"]].to_numpy().reshape(-1)
idx = vix_df.index.to_numpy().astype("datetime64[D]")
pd.DataFrame(vix_arr, index=idx, columns=["VIX"]).to_csv(f"./{output_dir}/VIX_{START.replace("-", "")}-{END.replace("-", "")}.csv", encoding='utf-8')

gspc_df = yf.download("^GSPC", start=START, end=END)
gspc_arr =gspc_df[["Close"]].to_numpy().reshape(-1)
idx = gspc_df.index.to_numpy().astype("datetime64[D]")
pd.DataFrame(gspc_arr, index=idx, columns=["GSPC"]).to_csv(f"./{output_dir}/GSPC_{START.replace("-", "")}-{END.replace("-", "")}.csv", encoding='utf-8')
