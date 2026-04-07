import re
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import time

txt_path = "./data_collection/S&P_500_component_stocks.txt"
output_dir = "data_collection"

def parsing_txt(txt_path):
    with open(txt_path, "r") as f:
        WIKI_TEXT = f.read()

    # ===== 1) 解析代號 =====
    # 支援三種樣式：{{NyseSymbol|XXX}}、{{NasdaqSymbol|YYY}}、{{BZX link|CBOE}}
    patterns = [
        r"\{\{\s*NyseSymbol\s*\|\s*([A-Za-z0-9\.\-\^]+)\s*\}\}",
        r"\{\{\s*NasdaqSymbol\s*\|\s*([A-Za-z0-9\.\-\^]+)\s*\}\}",
        r"\{\{\s*BZX\s+link\s*\|\s*([A-Za-z0-9\.\-\^]+)\s*\}\}",
    ]
    symbols = []
    for pat in patterns:
        symbols += re.findall(pat, WIKI_TEXT)

    clean = []
    for s in symbols:
        s = s.strip()
        s = re.sub(r"<!--.*?-->", "", s)
        s = re.sub(r"\s+", "", s)
        if s:
            clean.append(s)

    unique_symbols = sorted(set(clean))

    # 儲存原始解析出的代號
    pd.DataFrame({"Symbol": unique_symbols}).to_csv(
        f"./{output_dir}/sp500_symbols_from_wiki.csv", index=False, encoding="utf-8-sig"
    )

    # ===== 2) 轉成 Yahoo Finance 相容代號（. -> -）=====
    def to_yahoo_symbol(t: str) -> str:
        return t.replace(".", "-").strip()

    yahoo_syms = sorted({to_yahoo_symbol(s) for s in unique_symbols})

    pd.DataFrame({"YahooSymbol": yahoo_syms}).to_csv(
        f"./{output_dir}/sp500_symbols_yahoo.csv", index=False, encoding="utf-8-sig"
    )

    print(f"[INFO] 解析出 {len(unique_symbols)} 個代號，轉換後 {len(yahoo_syms)} 個 Yahoo 代號。")


# =================================================================================================

class Yahoo_Finance(object):
    def __init__(self):
        self.START = "2017-01-01"
        self.END = "2025-09-01"
        # self.END   = date.today()
        # self.START = self.END - timedelta(days=365*3 + 10)

        self.SYMBOLS_CSV =  f"./{output_dir}/sp500_symbols_yahoo.csv"
        self.PARQUET_OUT =  f"./{output_dir}/sp500_{self.START.replace("-", "")}-{self.END.replace("-", "")}_daily.parquet"
        self.CSV_OUT     =  f"./{output_dir}/sp500_{self.START.replace("-", "")}-{self.END.replace("-", "")}_daily.csv"
        self.CLOSE_OUT   =  f"./{output_dir}/sp500_{self.START.replace("-", "")}-{self.END.replace("-", "")}_close.csv"
        self.OPEN_OUT   =  f"./{output_dir}/sp500_{self.START.replace("-", "")}-{self.END.replace("-", "")}_open.csv"
        self.HIGH_OUT   =  f"./{output_dir}/sp500_{self.START.replace("-", "")}-{self.END.replace("-", "")}_high.csv"
        self.LOW_OUT   =  f"./{output_dir}/sp500_{self.START.replace("-", "")}-{self.END.replace("-", "")}_low.csv"
        self.VOLUME_OUT   =  f"./{output_dir}/sp500_{self.START.replace("-", "")}-{self.END.replace("-", "")}_volume.csv"

        self.MISSING_OUT =  f"./{output_dir}/sp500_{self.START.replace("-", "")}-{self.END.replace("-", "")}_missing_fields.csv"
        self.FAILED_OUT  =  f"./{output_dir}/sp500_{self.START.replace("-", "")}-{self.END.replace("-", "")}_failed_tickers.csv"

        self.SAVE_CSV   = True
        self.THREADS    = False
        self.BATCH_SIZE = 80
        self.MAX_RETRY  = 3
        self.SLEEP_BASE = 3.0

        self.FIELDS = ["Open","High","Low","Close","Adj Close","Volume"]

    def backoff_sleep(self, attempt, base= 3.0):
        time.sleep(base * (2 ** (attempt-1)))

    def download_batch(self, tickers):
        return yf.download(
            tickers=" ".join(tickers),
            start= self.START, end= self.END,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads= self.THREADS,
            progress=False,
        )

    def extract_fields(self, df, want_fields= None):
        """從批量結果取出指定欄位，回傳 MultiIndex 欄：(Ticker, Field)"""
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            lvl1 = set(df.columns.get_level_values(1))
            use = [f for f in want_fields if f in lvl1]
            if not use:
                return pd.DataFrame()
            sub = df.loc[:, df.columns.get_level_values(1).isin(use)].copy()
            sub = sub.sort_index(axis=1)
            return sub
        else:
            cols = [c for c in want_fields if c in df.columns]
            if not cols:
                return pd.DataFrame()
            sub = df[cols].copy()
            ticker = df.columns.name or "TICKER"
            sub.columns = pd.MultiIndex.from_product([[ticker], cols], names=[None, None])
            return sub

    def output(self, df):
        output_lst = [("Close", self.CLOSE_OUT), ("Open", self.OPEN_OUT), ("High", self.HIGH_OUT), ("Low", self.LOW_OUT), ("Volume", self.VOLUME_OUT)]
        for col, path in output_lst:
            if col in self.FIELDS:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df.xs(col, axis=1, level=1, drop_level=False).copy()
                    close.columns = close.columns.get_level_values(0)
                else:
                    close = pd.DataFrame()
                if not close.empty:
                    close.to_csv(path)
                    print(f"[完成] {col}-only -> {path}")

    def main(self):
        parsing_txt(txt_path)

        symbols = (
            pd.read_csv(self.SYMBOLS_CSV)["YahooSymbol"].dropna().astype(str).str.strip().tolist()
        )
        symbols = [s for s in symbols if s]
        print(f"[INFO] 標的數：{len(symbols)}")

        batches = [symbols[i:i+self.BATCH_SIZE] for i in range(0, len(symbols), self.BATCH_SIZE)]
        print(f"[INFO] 分批：{len(batches)} 批（每批約 {self.BATCH_SIZE} 檔）")

        parts = []
        failed = []

        for i, batch in enumerate(batches, 1):
            print(f"\n[批 {i}/{len(batches)}] 嘗試下載 {len(batch)} 檔")
            ok = False
            for attempt in range(1, self.MAX_RETRY+1):
                try:
                    raw = self.download_batch(batch)
                    sub = self.extract_fields(raw, self.FIELDS)
                    if not sub.empty:
                        parts.append(sub)
                        returned = set(sub.columns.get_level_values(0))
                        miss = [t for t in batch if t not in returned]
                        if miss:
                            failed.extend(miss)
                            print(f"  - 此批缺少 {len(miss)} 檔（已暫列失敗）")
                        print(f"  - 收到：{len(returned)} 檔，日期列：{sub.shape[0]}")
                        ok = True
                        break
                    else:
                        print(f"  - 空結果（第 {attempt} 次）")
                except Exception as e:
                    print(f"  - 例外（第 {attempt} 次）：{e}")
                if attempt < self.MAX_RETRY:
                    self.backoff_sleep(attempt, self.SLEEP_BASE)
            if not ok:
                failed.extend(batch)
                print("  - 此批最終失敗，已全數暫列為失敗")

        if not parts:
            print("\n[警告] 沒有任何成功批次，停止。")
        else:
            df_all = pd.concat(parts, axis=1)
            df_all = df_all.dropna(axis=1, how="all")
            df_all = df_all.loc[:, ~df_all.columns.duplicated(keep="first")]
            df_all = df_all.sort_index(axis=1)

            df_all.to_parquet(self.PARQUET_OUT)
            print(f"[完成] 寫出 {self.PARQUET_OUT}：{len(df_all.columns.get_level_values(0).unique())} 檔 × {len(df_all.columns.get_level_values(1).unique())} 欄 × {df_all.shape[0]} 天")

            if self.SAVE_CSV:
                flat = df_all.copy()
                flat.columns = [f"{t}_{f}" for t, f in flat.columns]
                flat.to_csv(self.CSV_OUT)
                print(f"[完成] 也寫出 {self.CSV_OUT}（寬表）")


            self.output(df_all)

            # if "Close" in self.FIELDS:
            #     if isinstance(df_all.columns, pd.MultiIndex):
            #         close = df_all.xs("Close", axis=1, level=1, drop_level=False).copy()
            #         close.columns = close.columns.get_level_values(0)
            #     else:
            #         close = pd.DataFrame()
            #     if not close.empty:
            #         close.to_csv(self.CLOSE_OUT)
            #         print(f"[完成] Close-only -> {self.CLOSE_OUT}")

            missing_rows = []
            tickers_in_df = sorted(df_all.columns.get_level_values(0).unique())
            for t in tickers_in_df:
                sub = df_all.loc[:, df_all.columns.get_level_values(0) == t]
                for f in self.FIELDS:
                    if f in sub.columns.get_level_values(1):
                        series = sub.xs((t, f), axis=1)
                        if series.dropna().empty:
                            missing_rows.append((t, f))
                    else:
                        missing_rows.append((t, f))
            pd.DataFrame(missing_rows, columns=["Ticker","Field"]).to_csv(self.MISSING_OUT, index=False)
            print(f"[完成] 缺欄位清單 -> {self.MISSING_OUT}（若為 0 列代表全部欄位都有數據）")

            # === 失敗名單 ===
            pd.DataFrame(sorted(set(failed)), columns=["YahooSymbol"]).to_csv(self.FAILED_OUT, index=False)
            print(f"[完成] 失敗名單 -> {self.FAILED_OUT}")

if __name__ == "__main__":
    yf_download = Yahoo_Finance()
    x = yf_download.main()
