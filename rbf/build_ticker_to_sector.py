# build_ticker_to_sector.py (hardened)
import pandas as pd
import numpy as np
import sys, re, time
from io import StringIO

LATENT_PATH = "./result/test_data_mapping_without_teacher_forcing.csv"
RET_PATH    = "./data/sp500_weekly_return_rate.csv"
OUT_CSV     = "ticker_to_sector.csv"

def _norm_ticker(x: str) -> str:
    x = str(x).strip().upper()
    x = x.replace("-", ".")
    x = re.sub(r"\s+", "", x)
    return x

def _to_yahoo_symbol(x: str) -> str:
    return x.replace(".", "-")

def load_needed_tickers(latent_path, ret_path):
    latent = pd.read_csv(latent_path, index_col=0)
    ret    = pd.read_csv(ret_path, index_col=0)
    lat = {_norm_ticker(i) for i in latent.index}
    cols = {_norm_ticker(c) for c in ret.columns}
    tickers = sorted({t for t in (lat | cols) if t and t != "NAN"})
    return tickers

def fetch_wikipedia_sp500():
    import requests
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print(f"[Try] Wikipedia: {url}")
    resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    # 用 StringIO 修復 FutureWarning
    tables = pd.read_html(StringIO(resp.text), flavor="bs4")
    df = tables[0].copy()
    df.columns = [c.strip().lower() for c in df.columns]
    sym_col = next((c for c in df.columns if "symbol" in c), None)
    sec_col = next((c for c in df.columns if "gics sector" in c or c == "sector"), None)
    if sym_col is None or sec_col is None:
        raise RuntimeError("Cannot find Symbol/GICS Sector on Wikipedia.")
    out = df[[sym_col, sec_col]].rename(columns={sym_col:"ticker", sec_col:"sector"})
    out["ticker"] = out["ticker"].map(_norm_ticker)
    out["sector"] = out["sector"].astype(str).str.strip()
    out = out.dropna().drop_duplicates(subset=["ticker"])
    print(f"[OK] Wikipedia rows: {len(out)}")
    return out, "Wikipedia"

def fetch_backup_list():
    # multiple backup：DataHub 兩個鏡像（任一可用就好）
    return [
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
        "https://pkgstore.datahub.io/core/s-and-p-500-companies/constituents/archive/constituents.csv",
    ]

def fetch_from_backup():
    errors = []
    for url in fetch_backup_list():
        try:
            print(f"[Try] Backup: {url}")
            df = pd.read_csv(url)
            df.columns = [c.strip().lower() for c in df.columns]
            sym_col = next((c for c in df.columns if c in ["symbol","ticker"]), None)
            sec_col = next((c for c in df.columns if c in ["sector","gics sector"]), None)
            if sym_col is None or sec_col is None:
                raise RuntimeError("No Symbol/Sector columns.")
            out = df[[sym_col, sec_col]].rename(columns={sym_col:"ticker", sec_col:"sector"})
            out["ticker"] = out["ticker"].map(_norm_ticker)
            out["sector"] = out["sector"].astype(str).str.strip()
            out = out.dropna().drop_duplicates(subset=["ticker"])
            print(f"[OK] Backup rows: {len(out)}")
            return out, url
        except Exception as e:
            errors.append(f"{url} -> {e}")
            print(f"[Warn] backup failed: {e}")
            continue
    raise RuntimeError("All backups failed:\n" + "\n".join(errors))

def fetch_yfinance_sectors(tickers):
    try:
        import yfinance as yf
    except ImportError:
        print("[Warn] yfinance not installed; skip this layer.")
        return pd.DataFrame(columns=["ticker","sector"]), "yfinance (skipped)"
    rows = []
    for t in tickers:
        ysym = _to_yahoo_symbol(t)
        sector = None
        for attempt in range(2):
            try:
                info = yf.Ticker(ysym).info
                sector = info.get("sector")
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(0.8)
                else:
                    print(f"[yfinance] fail {t}: {e}")
        rows.append((t, sector))
    out = pd.DataFrame(rows, columns=["ticker","sector"]).dropna(subset=["ticker"])
    out["sector"] = out["sector"].astype(str).str.strip()
    print(f"[OK] yfinance matched sectors: {out['sector'].notna().sum()} / {len(out)}")
    return out, "yfinance"

def main():
    try:
        needed = load_needed_tickers(LATENT_PATH, RET_PATH)
        print(f"[Info] needed tickers: {len(needed)}")
    except Exception as e:
        print(f"[Error] load files failed: {e}")
        sys.exit(1)

    df_all, src = None, None

    # 1) Wikipedia
    try:
        df_all, src = fetch_wikipedia_sp500()
    except Exception as e:
        print(f"[Warn] Wikipedia failed: {e}")

    # 2) Backups
    if df_all is None:
        try:
            df_all, src = fetch_from_backup()
        except Exception as e:
            print(f"[Warn] backups failed: {e}")

    # 3) yfinance
    if df_all is None:
        print("[Info] fallback to yfinance...")
        try:
            df_all, src = fetch_yfinance_sectors(needed)
        except Exception as e:
            print(f"[Warn] yfinance failed: {e}")

    # 4) 全失敗 → 空白模板
    if df_all is None or df_all.empty:
        print("[Fail] all sources failed. Output blank template for manual fill.")
        pd.DataFrame({"ticker": needed, "sector": [""]*len(needed)}).to_csv(OUT_CSV, index=False)
        print(f"[OUT] {OUT_CSV}")
        sys.exit(0)

    # 只保留我們需要的
    df_needed = df_all[df_all["ticker"].isin(needed)].copy()
    missing = sorted(set(needed) - set(df_needed["ticker"]))

    # 用 yfinance 補缺（若此時不是 yfinance 來源）
    if missing and src != "yfinance":
        print(f"[Info] still missing {len(missing)}; try yfinance to fill...")
        try:
            df_yf, _ = fetch_yfinance_sectors(missing)
            df_needed = pd.concat([df_needed, df_yf], ignore_index=True)\
                         .drop_duplicates(subset=["ticker"], keep="first")
            missing = sorted(set(needed) - set(df_needed["ticker"]))
        except Exception as e:
            print(f"[Warn] yfinance fill failed: {e}")

    df_needed = df_needed.sort_values("ticker")
    df_needed.to_csv(OUT_CSV, index=False)

    print(f"[OK] source: {src}")
    print(f"[OK] wrote: {OUT_CSV} ({len(df_needed)})")
    if missing:
        print(f"[Note] still missing ({len(missing)}): " + ", ".join(missing[:50]) + (" ..." if len(missing)>50 else ""))

if __name__ == "__main__":
    main()
