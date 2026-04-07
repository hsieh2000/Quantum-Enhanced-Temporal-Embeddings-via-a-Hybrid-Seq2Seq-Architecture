# QLSTM latent → Two RBF Strategies (RBF-DivMom & RBF-Graph) Backtest + Comparison
# Runnable directly in VS Code; outputs CSV and interactive Plotly HTML
# Optimized version: each date range outputs to its own folder; shared data loaded only once

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# -------------------- Path Constants --------------------
MODEL_LOG = "./models/model_logs.json"
RET_PATH = "./QLSTM_seq2seq/data/sp500_weekly_return_rate.csv"
RF_PATH = "./QLSTM_seq2seq/data/risk_free_weekly_rate.csv"
GSPC_PATH = "./QLSTM_seq2seq/data/gspc_weekly_return_rate.csv"
OUT_BASE_DIR = "./RBF/output"
LATENT_DIR = "./QLSTM_seq2seq/result"
SECTOR_CSV_PATH = "./RBF/ticker_to_sector.csv"

# -------------------- Visualization Parameters --------------------
TOP_N_HEATMAP = None   # e.g. 25 -> only plot top 25 weights; None -> all
TRIANGLE_MODE = "full" # "full" | "upper" | "lower"

# -------------------- Backtest Parameters --------------------
REBALANCE_EVERY = 4     # every 4 weeks ≈ monthly frequency
LOOK_MOM1 = 12          # 12-week momentum lookback
LOOK_MOM2 = 26          # 26-week momentum lookback
LOOK_COV  = 52          # 52-week covariance estimation window
N_SELECT  = 20          # RBF-DivMom: number of holdings
LAM       = 0.75        # RBF-DivMom: similarity penalty strength
M_TOP     = 80          # RBF-Graph: momentum pool size
ALPHA     = 5.0         # RBF-Graph: risk penalty
BETA      = 2.0         # RBF-Graph: similarity penalty

# -------------------- Utility Functions --------------------
def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv_vol = 1.0 / diag
    w = inv_vol / inv_vol.sum()
    return w

def rbf_diversified_momentum_select(mom_score: pd.Series, Ksub: pd.DataFrame, N: int = 20, lam: float = 0.75):
    names = mom_score.index.tolist()
    selected = []
    mof = mom_score.values.copy()
    for _ in range(min(N, len(names))):
        if selected:
            sim_to_sel = Ksub.values[:, selected].max(axis=1)
        else:
            sim_to_sel = np.zeros(len(names))
        obj = mof - lam * sim_to_sel
        obj[selected] = -1e9
        j = int(np.argmax(obj))
        selected.append(j)
    sel_names = [names[i] for i in selected]
    return sel_names

def projected_simplex(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0)
    s = w.sum()
    if s == 0:
        w = np.ones_like(w) / len(w)
    else:
        w /= s
    return w

def rbf_graph_opt_weights(exp_ret: pd.Series, cov: np.ndarray, Ksub: pd.DataFrame,
                          alpha=5.0, beta=2.0, iters=300, lr=0.03) -> pd.Series:
    cols = exp_ret.index.tolist()
    n = len(cols)
    w = np.ones(n) / n
    C = cov + 1e-8 * np.eye(n)
    G = Ksub.values
    r = exp_ret.values
    for _ in range(iters):
        grad = r - 2*alpha*(C @ w) - 2*beta*(G @ w)
        w = projected_simplex(w + lr * grad)
    return pd.Series(w, index=cols)

def perf_stats(weekly_ret: pd.Series) -> pd.Series:
    ann = 52
    r = weekly_ret.dropna()
    if len(r) < 5:
        return pd.Series({"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan})
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    mdd = dd.min()
    cagr = (eq.iloc[-1])**(ann/len(eq)) - 1
    vol = r.std() * np.sqrt(ann)
    sharpe = (r.mean() * ann) / (vol + 1e-12)
    return pd.Series({"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": float(mdd)})

def save_table_as_html(df: pd.DataFrame, title: str, path: str):
    fig = go.Figure(data=[go.Table(
        header=dict(values=["<b>"+c+"</b>" for c in df.columns],
                    fill_color="#1f2937", font_color="white", align="center"),
        cells=dict(values=[df[c] for c in df.columns], align="center")
    )])
    fig.update_layout(title=title)
    fig.write_html(path)

def _apply_triangle(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "full":
        return df
    m = df.copy()
    mask = np.triu(np.ones(m.shape), 1).astype(bool) if mode == "lower" else np.tril(np.ones(m.shape), -1).astype(bool)
    m.values[mask] = np.nan
    return m

def _save_similarity_heatmap(K_sub: pd.DataFrame, title: str, out_html: str):
    fig = px.imshow(
        K_sub,
        x=K_sub.columns, y=K_sub.index,
        color_continuous_scale="Viridis",
        title=title,
        aspect="auto",
        zmin=0.0, zmax=1.0
    )
    fig.update_xaxes(side="top", tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.update_traces(hovertemplate="i=%{y}<br>j=%{x}<br>similarity=%{z:.3f}<extra></extra>")
    fig.update_layout(margin=dict(l=60, r=20, t=60, b=60))
    fig.write_html(out_html)

def _make_and_save_heatmap(K: pd.DataFrame, weights_csv: str, raw_title: str, out_html: str):
    if not os.path.exists(weights_csv):
        print(f"[Skip] {weights_csv} not found.")
        return
    w = pd.read_csv(weights_csv, index_col=0)["weight"]
    names = [t for t in w.sort_values(ascending=False).index if t in K.index]
    if TOP_N_HEATMAP is not None:
        names = names[:min(TOP_N_HEATMAP, len(names))]
    K_sub = K.loc[names, names].copy()
    K_sub = _apply_triangle(K_sub, TRIANGLE_MODE)
    _save_similarity_heatmap(K_sub, raw_title, out_html)

def load_common_data():
    """Load shared data (only needs to run once)"""
    print("Loading common data (returns, risk-free rate, benchmark)...")
    ret = pd.read_csv(RET_PATH, index_col=0)
    ret.index = pd.to_datetime(ret.index)

    rf = pd.read_csv(RF_PATH, index_col=0).iloc[:, 0]
    rf.index = pd.to_datetime(rf.index)

    gspc = pd.read_csv(GSPC_PATH, index_col=0).iloc[:, 0]
    gspc.index = pd.to_datetime(gspc.index)

    # Align timestamps
    common_weeks = ret.index.intersection(rf.index).intersection(gspc.index)
    ret = ret.loc[common_weeks].astype(float)
    rf = rf.loc[common_weeks].astype(float)
    gspc = gspc.loc[common_weeks].astype(float)

    return ret, rf, gspc

def load_latent_and_align(latent_path, ret):
    """Load latent representations and align with return data"""
    latent = pd.read_csv(latent_path, index_col=0).dropna()

    # Align ticker symbols
    tickers = latent.index.intersection(ret.columns)
    latent = latent.loc[tickers]
    ret_aligned = ret[tickers].sort_index()

    return latent, ret_aligned, tickers

def build_rbf_kernel(latent: pd.DataFrame, tickers, out_dir):
    """Build the RBF kernel matrix"""
    X = latent.values
    D = squareform(pdist(X, metric="euclidean"))
    med = np.median(D[np.triu_indices_from(D, 1)])
    sigma = med / np.sqrt(2) if med > 0 else 1e-6
    K = np.exp(-(D**2) / (2 * sigma**2))
    K = pd.DataFrame(K, index=tickers, columns=tickers)
    K.to_csv(os.path.join(out_dir, "rbf_kernel.csv"))
    return K

# -------------------- Backtest Function --------------------

def run_backtest(ret: pd.DataFrame, K: pd.DataFrame, start_date: str, end_date: str):
    """
    Run backtest. ret must contain LOOK_COV weeks of history before ts_start.
    Only generates returns within [ts_start, ts_end).
    """
    ts_start_dt = pd.to_datetime(start_date)
    ts_end_dt = pd.to_datetime(end_date)

    dates = ret.index  # Use the full time index

    # 1. Find all potential rebalance point indices
    # i is the index of the first week of the holding period (the date weights take effect)
    all_rebal_idx = list(range(LOOK_COV, len(dates), REBALANCE_EVERY))

    # 2. Filter rebalance points: holding period must fall within [ts_start, ts_end)
    rebal_idx = []
    for i in all_rebal_idx:
        hold_start = dates[i]

        # Condition 1: first week of holding (dates[i]) must be >= ts_start_dt
        if hold_start >= ts_start_dt:
            # Condition 2: holding period start must be before ts_end
            if hold_start < ts_end_dt:
                rebal_idx.append(i)

    # Check if there is sufficient data for backtesting
    if len(rebal_idx) == 0:
        # This catches cases where the first interval (e.g. 2022-04-01 to 2022-07-01) lacks
        # enough LOOK_COV history, so the first rebalance point is far before ts_start_dt.
        print(f"Warning: No effective rebalance points found in [{start_date}, {end_date})!")
        if len(dates) >= LOOK_COV:
            print(f"      - First possible rebalance date (index {LOOK_COV}): {dates[LOOK_COV]}")
        else:
            print(f"      - Total available data points: {len(dates)}. Not even enough history for 1st rebalance.")
        return [], [], None, None

    print(f"Effective Rebalance Points: {len(rebal_idx)}")

    ts_div_list, ts_graph_list = [], []
    last_w_div, last_w_graph = None, None

    for i in rebal_idx:
        # History window ends at dates[i-1]
        end = dates[i-1]

        # Historical data: always use LOOK_COV weeks of history
        hist = ret.loc[dates[i-LOOK_COV]:end]

        # Momentum calculation: always use LOOK_MOM weeks of history
        mom12 = (1 + ret.loc[dates[i-LOOK_MOM1]:end]).prod() - 1
        mom26 = (1 + ret.loc[dates[i-LOOK_MOM2]:end]).prod() - 1
        mom_score = (0.6 * mom12 + 0.4 * mom26).dropna()

        live = mom_score.index.intersection(K.index)
        mom_score = mom_score.loc[live]

        # Strategy 1: RBF-DivMom
        sel = rbf_diversified_momentum_select(mom_score, K.loc[live, live], N=N_SELECT, lam=LAM)
        sub_hist = hist[sel].dropna()
        if sub_hist.shape[1] >= 5:
            cov = sub_hist.cov().values
            w_rp = risk_parity_weights(cov)
            tilt = mom_score[sel].values
            tilt = tilt / (tilt.sum() + 1e-12)
            w_div = (0.5 * w_rp + 0.5 * tilt)
            w_div = w_div / w_div.sum()
            last_w_div = pd.Series(w_div, index=sub_hist.columns)
        else:
            last_w_div = None

        # Strategy 2: RBF-Graph
        topM = mom_score.sort_values(ascending=False).head(min(M_TOP, len(mom_score))).index.tolist()
        sub_hist2 = hist[topM].dropna()
        if sub_hist2.shape[1] >= 5:
            cov2 = sub_hist2.cov().values
            exp_r = sub_hist2.mean() * 52.0
            w_graph = rbf_graph_opt_weights(exp_r, cov2, K.loc[sub_hist2.columns, sub_hist2.columns],
                                            alpha=ALPHA, beta=BETA, iters=300, lr=0.03)
            last_w_graph = w_graph.copy()
        else:
            last_w_graph = None

        # Forward holding returns (i is the index of the first week of the holding period)
        hold_slice = ret.iloc[i:i+REBALANCE_EVERY]

        # 3. Strictly restrict holding period to [ts_start_dt, ts_end_dt)
        hold_slice = hold_slice.loc[(hold_slice.index >= ts_start_dt) & (hold_slice.index < ts_end_dt)]

        if not hold_slice.empty:
            if last_w_div is not None:
                valid_assets = last_w_div.index.intersection(hold_slice.columns)
                w_div_aligned = last_w_div.loc[valid_assets]
                r1 = (hold_slice[valid_assets] @ w_div_aligned.values).fillna(0)
                ts_div_list.append(r1)

            if last_w_graph is not None:
                valid_assets = last_w_graph.index.intersection(hold_slice.columns)
                w_graph_aligned = last_w_graph.loc[valid_assets]
                r2 = (hold_slice[valid_assets] * w_graph_aligned.values).sum(axis=1).fillna(0)
                ts_graph_list.append(r2)

    return ts_div_list, ts_graph_list, last_w_div, last_w_graph

def save_results(ts_div_list, ts_graph_list, last_w_div, last_w_graph, rf, gspc, out_dir, cumulative_values=None):
    """Save backtest results (supports continuous equity curves)

    Args:
        cumulative_values: dict containing ending equity values from the previous period,
                           e.g. {"RBF_DivMom": 1.25, "RBF_Graph": 1.18, "SP500": 1.15}
    """
    ts_div = pd.concat(ts_div_list).sort_index() if ts_div_list else pd.Series(dtype=float)
    ts_graph = pd.concat(ts_graph_list).sort_index() if ts_graph_list else pd.Series(dtype=float)
    ts_bench = gspc.loc[ts_div.index] if len(ts_div) else pd.Series(dtype=float)

    # Handle empty data case
    if cumulative_values is None:
        cumulative_values = {"RBF_DivMom": 1.0, "RBF_Graph": 1.0, "SP500": 1.0}

    # Compute equity curves
    if len(ts_div) > 0:
        # Has backtest data: start from cumulative value
        eq_div = cumulative_values["RBF_DivMom"] * (1 + ts_div).cumprod()
        eq_graph = cumulative_values["RBF_Graph"] * (1 + ts_graph).cumprod()
        eq_bench = cumulative_values["SP500"] * (1 + ts_bench).cumprod()

        # Update cumulative values for the next period
        new_cumulative_values = {
            "RBF_DivMom": eq_div.iloc[-1],
            "RBF_Graph": eq_graph.iloc[-1],
            "SP500": eq_bench.iloc[-1]
        }
    else:
        # No backtest data: keep cumulative values unchanged
        eq_div = pd.Series(dtype=float)
        eq_graph = pd.Series(dtype=float)
        eq_bench = pd.Series(dtype=float)
        new_cumulative_values = cumulative_values.copy()
        print(f"  ℹ️  No backtest data - cumulative values remain: {new_cumulative_values}")

    eq = pd.DataFrame({"RBF_DivMom": eq_div, "RBF_Graph": eq_graph, "SP500": eq_bench}).dropna(how="all")
    eq.to_csv(os.path.join(out_dir, "equity_curves.csv"))

    # Metrics (computed on current-period returns only)
    ex_div = ts_div - rf.loc[ts_div.index] if len(ts_div) > 0 else pd.Series(dtype=float)
    ex_graph = ts_graph - rf.loc[ts_graph.index] if len(ts_graph) > 0 else pd.Series(dtype=float)
    ex_bench = ts_bench - rf.loc[ts_bench.index] if len(ts_bench) > 0 else pd.Series(dtype=float)

    metrics = pd.concat([
        perf_stats(ex_div).rename("RBF-DivMom (excess)"),
        perf_stats(ex_graph).rename("RBF-Graph (excess)"),
        perf_stats(ex_bench).rename("S&P 500 (excess)")
    ], axis=1).round(4)
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"))

    # Portfolio weight output
    if last_w_div is not None:
        last_w_div.sort_values(ascending=False).to_frame("weight").to_csv(
            os.path.join(out_dir, "rbf_portfolio_divmom_latest.csv"))
    if last_w_graph is not None:
        last_w_graph.sort_values(ascending=False).to_frame("weight").to_csv(
            os.path.join(out_dir, "rbf_portfolio_graph_latest.csv"))

    return eq, new_cumulative_values, metrics

def generate_latent_by_sector_plot(embedding_df: pd.DataFrame, sector_csv_path: str, out_path: str, label: str):
    """
    Read sector data and generate a scatter plot of the latent space colored by sector.
    """
    if not os.path.exists(sector_csv_path):
        print(f"[Sector Plot] Warning: Sector file not found at {sector_csv_path}. Skipping plot generation for {label}.")
        return

    # 1. Read and normalize sector data
    try:
        sec = pd.read_csv(sector_csv_path)
        sec.columns = [c.strip().lower() for c in sec.columns]
        if not {"ticker", "sector"}.issubset(sec.columns):
            print("[Sector Plot] Error: Sector CSV must contain 'ticker' and 'sector' columns. Skipping.")
            return
        sec["ticker"] = sec["ticker"].astype(str).str.strip().str.upper()
        sec["sector"] = sec["sector"].astype(str).str.strip()
        sec = sec.set_index("ticker")
    except Exception as e:
        print(f"[Sector Plot] Error reading or processing sector CSV: {e}. Skipping.")
        return

    # 2. Align with embedding
    Z = embedding_df.copy()
    # Ensure latent index is also uppercased and stripped (consistent with load_latent_and_align)
    Z.index = Z.index.astype(str).str.strip().str.upper()
    common = Z.index.intersection(sec.index)

    # Ensure Z has at least 'f1', 'f2' columns
    if len(Z.columns) < 2:
        print("[Sector Plot] Error: Latent data must have at least 2 features (f1, f2). Skipping.")
        return

    Z = Z.loc[common, Z.columns[:2]].astype(float)  # Assume first two columns are f1, f2
    lab = sec.loc[common, "sector"]

    # 3. Filter invalid data
    mask_valid = (Z.notna().all(axis=1) & lab.notna() & (lab.astype(str).str.len() > 0))
    Z = Z.loc[mask_valid]
    lab = lab.loc[mask_valid]

    if Z.shape[0] < 2:
        print(f"[Sector Plot] Warning: Insufficient valid data points ({Z.shape[0]}) after alignment with sectors. Skipping.")
        return

    # 4. Generate scatter plot
    df_plot = Z.copy()
    df_plot["Sector"] = lab.values
    df_plot.columns = ['f1', 'f2', 'Sector']

    num_tickers = Z.shape[0]
    num_sectors = len(pd.unique(lab))
    title_text = f"Latent Map by Sector ({label}) | {num_tickers} Tickers, {num_sectors} Sectors"

    fig_scatter = px.scatter(df_plot, x="f1", y="f2", color="Sector",
                             hover_name=df_plot.index,
                             title=title_text)

    output_file = os.path.join(out_path, f"plotly_latent_by_sector_{label}.html")
    fig_scatter.write_html(output_file)
    print(f"  ✓ Sector latent map saved: {output_file}")


def create_visualizations(latent, K, eq, out_dir, period_label):
    """Create all visualization charts"""
    # Latent space scatter plot
    density = K.sum(axis=1)
    df_scatter = latent.copy()
    df_scatter["rbf_density"] = density.loc[df_scatter.index].values
    fig_latent = px.scatter(df_scatter, x="f1", y="f2",
                            color="rbf_density", color_continuous_scale="Viridis",
                            hover_name=df_scatter.index,
                            title="QLSTM latent map (colored by RBF density)")
    fig_latent.write_html(os.path.join(out_dir, "plotly_latent_map.html"))

    # Generate sector plot for current latent data
    # ----------------------------------------------------
    latent_for_sector = latent.copy()
    latent_for_sector.columns = ['f1', 'f2']  # Ensure column names are correct
    generate_latent_by_sector_plot(
        embedding_df=latent_for_sector,
        sector_csv_path=SECTOR_CSV_PATH,
        out_path=out_dir,
        label=period_label  # Use current period as label
    )
    # ----------------------------------------------------

    # Similarity heatmaps
    _make_and_save_heatmap(
        K,
        os.path.join(out_dir, "rbf_portfolio_divmom_latest.csv"),
        f"RBF similarity – Selected (RBF-DivMom) | triangle={TRIANGLE_MODE} | topN={TOP_N_HEATMAP}",
        os.path.join(out_dir, "plotly_rbf_similarity_divmom.html")
    )
    _make_and_save_heatmap(
        K,
        os.path.join(out_dir, "rbf_portfolio_graph_latest.csv"),
        f"RBF similarity – Selected (RBF-Graph) | triangle={TRIANGLE_MODE} | topN={TOP_N_HEATMAP}",
        os.path.join(out_dir, "plotly_rbf_similarity_graph.html")
    )

    # Equity curve chart
    fig_eq = go.Figure()
    if "RBF_DivMom" in eq.columns:
        fig_eq.add_trace(go.Scatter(x=eq.index, y=eq["RBF_DivMom"], mode='lines', name="RBF-DivMom"))
    if "RBF_Graph" in eq.columns:
        fig_eq.add_trace(go.Scatter(x=eq.index, y=eq["RBF_Graph"], mode='lines', name="RBF-Graph"))
    if "SP500" in eq.columns:
        fig_eq.add_trace(go.Scatter(x=eq.index, y=eq["SP500"], mode='lines', name="S&P 500"))
    fig_eq.update_layout(title="Equity Curves (weekly compounded)",
                         xaxis_title="Date", yaxis_title="Net Value", legend=dict(x=0.01, y=0.99))
    fig_eq.write_html(os.path.join(out_dir, "plotly_equity_curves.html"))

def main():
    """Main function"""
    # Load model logs
    with open(MODEL_LOG, "r", encoding="utf-8") as f:
        model_log = json.load(f)

    # Load shared data only once
    ret_full, rf, gspc = load_common_data()

    # List all latent CSV files
    # csv_files = [f for f in os.listdir(LATENT_DIR) if f.startswith("training_latent_data") and f.endswith(".csv")]
    csv_files = [f for f in os.listdir(LATENT_DIR) if f.startswith("test_latent_data") and f.endswith(".csv")]

    if len(csv_files) == 0:
        print(f"No latent CSV files found in {LATENT_DIR}")
        return

    # Track cumulative equity values for each strategy (used for continuous calculation)
    cumulative_values = {
        "RBF_DivMom": 1.0,
        "RBF_Graph": 1.0,
        "SP500": 1.0
    }

    # Store equity curves from all periods (used for merging)
    all_equity_curves = []
    all_metrics = []

    # Run backtest for each latent file
    for idx, csv_file in enumerate(csv_files):
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(csv_files)}: {csv_file}")
        print(f"{'='*60}")

        latent_path = os.path.join(LATENT_DIR, csv_file)

        # Get the corresponding date range from model logs
        if idx < len(model_log.get("Seq2seq_models", [])):
            model_info = model_log["Seq2seq_models"][idx]
            ts_start = model_info["testing range"]["start"]
            ts_end = model_info["testing range"]["end"]
            date_folder = f"{ts_start}_to_{ts_end}".replace("-", "")
        else:
            date_folder = f"model_{idx+1}"

        # Create output directory for this date range
        out_dir = os.path.join(OUT_BASE_DIR, date_folder)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output directory: {out_dir}")

        try:
            # Load latent and align
            print(f"Loading latent from {csv_file}...")
            latent, ret_aligned, tickers = load_latent_and_align(latent_path, ret_full)

            # Build RBF kernel matrix
            print("Building RBF kernel...")
            K = build_rbf_kernel(latent, tickers, out_dir)

            # Run backtest
            print(f"Running backtest for period {ts_start} to {ts_end}...")
            ts_div_list, ts_graph_list, last_w_div, last_w_graph = run_backtest(
                ret_aligned, K, start_date=ts_start, end_date=ts_end
            )

            # Save results (pass cumulative values to maintain continuity)
            print("Saving results...")
            eq, cumulative_values, metrics = save_results(
                ts_div_list, ts_graph_list, last_w_div, last_w_graph,
                rf, gspc, out_dir, cumulative_values
            )

            # Store current period data (for final merge) - only append if data exists
            if len(eq) > 0:
                eq['Period'] = date_folder
                all_equity_curves.append(eq)

            metrics_with_period = metrics.T
            metrics_with_period['Period'] = date_folder
            all_metrics.append(metrics_with_period)

            # Create visualizations
            print("Creating visualizations...")
            create_visualizations(latent, K, eq, out_dir, date_folder)

            # Results summary
            print(f"\n✓ Completed for {date_folder}")
            print(f"  - Metrics: {os.path.join(out_dir, 'metrics.csv')}")
            print(f"  - Equity curves: {os.path.join(out_dir, 'equity_curves.csv')}")
            print(f"  - Visualizations: {out_dir}")

        except Exception as e:
            print(f"\n✗ Error processing {csv_file}: {str(e)}")
            continue

    # ==================== Merge All Period Outputs ====================
    if all_equity_curves and all_metrics:
        print(f"\n{'='*60}")
        print("Creating combined outputs...")
        print(f"{'='*60}")

        # 1. Merge equity curves (already continuous)
        combined_eq = pd.concat(all_equity_curves).drop(columns=['Period'])
        combined_eq = combined_eq[~combined_eq.index.duplicated(keep='first')]  # Remove duplicate dates
        combined_eq = combined_eq.sort_index()

        # Check if there is sufficient data
        if len(combined_eq) < 2:
            print(f"  ⚠ Warning: Not enough data for combined analysis (only {len(combined_eq)} rows)")
        else:
            combined_eq.to_csv(os.path.join(OUT_BASE_DIR, "combined_equity_curves.csv"))
            print(f"  ✓ Combined equity curves: {os.path.join(OUT_BASE_DIR, 'combined_equity_curves.csv')}")

            # 2. Merge performance metrics
            combined_metrics = pd.concat(all_metrics)
            combined_metrics = combined_metrics.reset_index()
            combined_metrics = combined_metrics.rename(columns={'index': 'Metric'})
            # Reorder columns: put Period first
            cols = ['Period'] + [c for c in combined_metrics.columns if c != 'Period']
            combined_metrics = combined_metrics[cols]
            combined_metrics.to_csv(os.path.join(OUT_BASE_DIR, "summary_all_periods_metrics.csv"), index=False)
            print(f"  ✓ Summary metrics: {os.path.join(OUT_BASE_DIR, 'summary_all_periods_metrics.csv')}")

            # 3. Generate combined equity curve chart
            fig_combined = go.Figure()
            if "RBF_DivMom" in combined_eq.columns:
                fig_combined.add_trace(go.Scatter(x=combined_eq.index, y=combined_eq["RBF_DivMom"],
                                                  mode='lines', name="RBF-DivMom", line=dict(width=2)))
            if "RBF_Graph" in combined_eq.columns:
                fig_combined.add_trace(go.Scatter(x=combined_eq.index, y=combined_eq["RBF_Graph"],
                                                  mode='lines', name="RBF-Graph", line=dict(width=2)))
            if "SP500" in combined_eq.columns:
                fig_combined.add_trace(go.Scatter(x=combined_eq.index, y=combined_eq["SP500"],
                                                  mode='lines', name="S&P 500", line=dict(width=2, dash='dash')))

            fig_combined.update_layout(
                title="Combined Equity Curves (Continuous across all periods)",
                xaxis_title="Date",
                yaxis_title="Cumulative Net Value",
                legend=dict(x=0.01, y=0.99),
                hovermode='x unified'
            )
            fig_combined.write_html(os.path.join(OUT_BASE_DIR, "combined_equity_curves.html"))
            print(f"  ✓ Combined visualization: {os.path.join(OUT_BASE_DIR, 'combined_equity_curves.html')}")

            # 4. Compute overall performance statistics (with safety checks)
            print(f"\n  📊 Overall Performance:")
            if "RBF_DivMom" in combined_eq.columns and len(combined_eq["RBF_DivMom"].dropna()) >= 2:
                total_return_div = (combined_eq["RBF_DivMom"].iloc[-1] / combined_eq["RBF_DivMom"].iloc[0]) - 1
                print(f"     RBF-DivMom:  {total_return_div:>8.2%}")
            else:
                print(f"     RBF-DivMom:  N/A (insufficient data)")

            if "RBF_Graph" in combined_eq.columns and len(combined_eq["RBF_Graph"].dropna()) >= 2:
                total_return_graph = (combined_eq["RBF_Graph"].iloc[-1] / combined_eq["RBF_Graph"].iloc[0]) - 1
                print(f"     RBF-Graph:   {total_return_graph:>8.2%}")
            else:
                print(f"     RBF-Graph:   N/A (insufficient data)")

            if "SP500" in combined_eq.columns and len(combined_eq["SP500"].dropna()) >= 2:
                total_return_sp500 = (combined_eq["SP500"].iloc[-1] / combined_eq["SP500"].iloc[0]) - 1
                print(f"     S&P 500:     {total_return_sp500:>8.2%}")
            else:
                print(f"     S&P 500:     N/A (insufficient data)")
    else:
        print(f"\n  ⚠ No data to combine (all_equity_curves: {len(all_equity_curves)}, all_metrics: {len(all_metrics)})")

    print(f"\n{'='*60}")
    print("All backtests completed!")
    print(f"Individual period results: {OUT_BASE_DIR}/[date_folders]")
    print(f"Combined results: {OUT_BASE_DIR}/combined_*")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()