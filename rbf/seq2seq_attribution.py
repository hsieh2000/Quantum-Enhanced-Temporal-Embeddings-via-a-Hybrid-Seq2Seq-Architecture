# Seq2Seq → Rigorous Validation of Whether RBF Portfolio Performance Originates from Seq2Seq
# Three-layer validation: Fidelity / Structure / Attribution + (Extended) Sector-based Tests (cross-embedding comparison)
# Runnable directly in VS Code; outputs CSV and interactive Plotly HTML

import os
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import plotly.express as px
import plotly.graph_objects as go

# -------------------- Paths --------------------
TF_CSV   = "./QLSTM_seq2seq/result/test_data_mapping_with_teacher_forcing.csv"
FR_CSV   = "./QLSTM_seq2seq/result/test_data_mapping_without_teacher_forcing.csv"
RET_CSV  = "./QLSTM_seq2seq/data/sp500_weekly_return_rate.csv"
RF_CSV   = "./QLSTM_seq2seq/data/risk_free_weekly_rate.csv"
GSPC_CSV = "./QLSTM_seq2seq/data/gspc_weekly_return_rate.csv"
SECTOR_CSV = "./RBF/ticker_to_sector.csv"  # Sector validation is skipped automatically if this file is missing

OUT = "./RBF/output"
os.makedirs(OUT, exist_ok=True)

# -------------------- Base Data --------------------
tf = pd.read_csv(TF_CSV, index_col=0)
fr = pd.read_csv(FR_CSV, index_col=0)
ret = pd.read_csv(RET_CSV, index_col=0);          ret.index = pd.to_datetime(ret.index)
rf  = pd.read_csv(RF_CSV, index_col=0).iloc[:,0]; rf.index  = pd.to_datetime(rf.index)
gspc= pd.read_csv(GSPC_CSV, index_col=0).iloc[:,0]; gspc.index = pd.to_datetime(gspc.index)

# Align ticker universe (Free-run as the reference)
tickers = fr.index.intersection(ret.columns)
fr = fr.loc[tickers].astype(float)
tf = tf.loc[tickers].astype(float)
ret = ret[tickers].sort_index()

# Align timestamps
common_weeks = ret.index.intersection(rf.index).intersection(gspc.index)
ret  = ret.loc[common_weeks].astype(float)
rf   = rf.loc[common_weeks].astype(float)
gspc = gspc.loc[common_weeks].astype(float)

# -------------------- 1) Fidelity: TF vs Free-run → Representation Quality --------------------
# (a) Average distance between the two mapping point sets
map_gap = np.mean(np.linalg.norm(tf.values - fr.values, axis=1))
# (b) MSE_TF / MSE_Free can also be read from model logs for comparison if available
pd.Series({"mapping_avg_distance": map_gap}).to_csv(os.path.join(OUT, "fidelity_metrics.csv"))

# Interactive scatter: TF vs Free-run overlay (check for overlap)
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(x=tf["f1"], y=tf["f2"], mode="markers",
                                 name="Teacher Forcing", text=tf.index))
fig_scatter.add_trace(go.Scatter(x=fr["f1"], y=fr["f2"], mode="markers",
                                 name="Free-run", text=fr.index))
fig_scatter.update_layout(title=f"Latent mappings (TF vs Free-run) | avg dist={map_gap:.4f}",
                          xaxis_title="f1", yaxis_title="f2")
fig_scatter.write_html(os.path.join(OUT, "latent_tf_vs_fr.html"))

# -------------------- 2) Structure: Latent Distance vs Behavioral Distance (historical correlation) --------------------
# Latent distance (using Free-run)
D_lat = squareform(pdist(fr.values, metric="euclidean"))
# Behavioral distance: 1 - corr (high correlation → small distance)
C = ret.corr().values
D_beh = 1.0 - C
# Flatten to vectors (upper triangle)
tri = np.triu_indices_from(D_lat, k=1)
v_lat = D_lat[tri]
v_beh = D_beh[tri]
# Correlation between distance vectors (Spearman; can also use distance correlation / HSIC)
rho, pval = spearmanr(v_lat, v_beh)
pd.Series({"spearman_lat_vs_behavior": rho, "p_value": pval}).to_csv(os.path.join(OUT, "structure_metrics.csv"))

# Interactive heatmaps: both distance matrices
fig_lat = px.imshow(pd.DataFrame(D_lat, index=tickers, columns=tickers),
                    title="Latent distance (Free-run, Euclidean)", aspect="auto", color_continuous_scale="Viridis")
fig_lat.update_xaxes(side="top", tickangle=45)
fig_lat.write_html(os.path.join(OUT, "latent_distance.html"))

fig_beh = px.imshow(pd.DataFrame(D_beh, index=tickers, columns=tickers),
                    title="Behavior distance (1 - corr of weekly returns)", aspect="auto", color_continuous_scale="Viridis")
fig_beh.update_xaxes(side="top", tickangle=45)
fig_beh.write_html(os.path.join(OUT, "behavior_distance.html"))

# -------------------- 3) Attribution: Swap Embeddings → Same RBF Portfolio → Compare Performance --------------------
def rbf_kernel(xy: np.ndarray, sigma: float = None):
    D = squareform(pdist(xy, metric="euclidean"))
    if sigma is None:
        med = np.median(D[np.triu_indices_from(D,1)])
        sigma = (med / np.sqrt(2)) if med > 0 else 1e-6
    K = np.exp(-(D**2)/(2*sigma**2))
    return pd.DataFrame(K, index=tickers, columns=tickers), sigma

def risk_parity_weights(cov):
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    w = (1.0 / d); w /= w.sum()
    return w

def rbf_diverse_select(mom_score: pd.Series, Ksub: pd.DataFrame, N=20, lam=0.75):
    names = mom_score.index.tolist()
    sel = []
    v = mom_score.values.copy()
    for _ in range(min(N,len(names))):
        if sel:
            sim = Ksub.values[:, sel].max(axis=1)
        else:
            sim = np.zeros(len(names))
        obj = v - lam*sim
        obj[sel] = -1e9
        j = int(np.argmax(obj)); sel.append(j)
    return [names[j] for j in sel]

def proj_simplex(w):
    w = np.maximum(w,0); s=w.sum()
    return w/s if s>0 else np.ones_like(w)/len(w)

def rbf_graph_opt(exp_ret: pd.Series, cov, Ksub: pd.DataFrame, alpha=5.0, beta=2.0, iters=300, lr=0.03):
    cols = exp_ret.index.tolist()
    n = len(cols)
    w = np.ones(n)/n
    C = cov + 1e-8*np.eye(n)
    G = Ksub.values
    r = exp_ret.values
    for _ in range(iters):
        grad = r - 2*alpha*(C@w) - 2*beta*(G@w)
        w = proj_simplex(w + lr*grad)
    return pd.Series(w, index=cols)

def perf_stats(weekly_ret: pd.Series):
    ann=52; r=weekly_ret.dropna()
    if len(r)<5: return pd.Series({"CAGR":np.nan,"Vol":np.nan,"Sharpe":np.nan,"MaxDD":np.nan})
    eq=(1+r).cumprod(); peak=eq.cummax(); dd=(eq-peak)/peak; mdd=dd.min()
    cagr=eq.iloc[-1]**(ann/len(eq))-1; vol=r.std()*np.sqrt(ann); sharpe=(r.mean()*ann)/(vol+1e-12)
    return pd.Series({"CAGR":cagr,"Vol":vol,"Sharpe":sharpe,"MaxDD":float(mdd)})

def run_backtest_with_embedding(embedding_2d: pd.DataFrame,
                                label: str,
                                N_select=20, lam=0.75, M_top=80, alpha=5.0, beta=2.0,
                                rebalance_every=4, look_mom1=12, look_mom2=26, look_cov=52):
    # Build RBF kernel from embedding
    K, _ = rbf_kernel(embedding_2d.loc[tickers].values)
    dates = ret.index
    rebal_idx = list(range(look_cov, len(dates)-rebalance_every, rebalance_every))
    ts_div, ts_graph = [], []
    last_w_div, last_w_graph = None, None

    for i in rebal_idx:
        end = dates[i]; hist=ret.loc[dates[i-look_cov]:end]
        mom12=(1+ret.loc[dates[i-look_mom1]:end]).prod()-1
        mom26=(1+ret.loc[dates[i-look_mom2]:end]).prod()-1
        mom  =(0.6*mom12+0.4*mom26).dropna()
        live = mom.index.intersection(K.index)
        mom  = mom.loc[live]
        # Strategy 1: select N stocks + risk-parity / momentum tilt
        sel = rbf_diverse_select(mom, K.loc[live, live], N=N_select, lam=lam)
        sub1= hist[sel].dropna()
        if sub1.shape[1]>=5:
            cov=sub1.cov().values; w_rp=risk_parity_weights(cov)
            tilt=(mom[sel].values); tilt/= (tilt.sum()+1e-12)
            w_div = proj_simplex(0.5*w_rp + 0.5*tilt)
            last_w_div = pd.Series(w_div, index=sub1.columns)
        # Strategy 2: continuous weight optimization
        topM = mom.sort_values(ascending=False).head(min(M_top,len(mom))).index.tolist()
        sub2= hist[topM].dropna()
        if sub2.shape[1]>=5:
            cov2=sub2.cov().values; exp_r=sub2.mean()*52.0
            w_graph = rbf_graph_opt(exp_r, cov2, K.loc[sub2.columns, sub2.columns], alpha=alpha, beta=beta)
            last_w_graph = w_graph.copy()
        hold = ret.iloc[i+1:i+1+rebalance_every]
        if last_w_div is not None:
            ts_div.append((hold[last_w_div.index]@last_w_div.values).fillna(0))
        if last_w_graph is not None:
            ts_graph.append((hold[last_w_graph.index]*last_w_graph.values).sum(axis=1).fillna(0))
    ts_div   = pd.concat(ts_div).sort_index()   if ts_div else pd.Series(dtype=float)
    ts_graph = pd.concat(ts_graph).sort_index() if ts_graph else pd.Series(dtype=float)
    ts_bench = gspc.loc[ts_div.index]
    # Metrics (excess over risk-free rate)
    ex_div   = ts_div   - rf.loc[ts_div.index]
    ex_graph = ts_graph - rf.loc[ts_graph.index]
    ex_bench = ts_bench - rf.loc[ts_bench.index]
    metrics = pd.concat([
        perf_stats(ex_div).rename("RBF-DivMom (excess)"),
        perf_stats(ex_graph).rename("RBF-Graph (excess)"),
        perf_stats(ex_bench).rename("S&P 500 (excess)")
    ], axis=1)
    # Equity curves
    eq = pd.DataFrame({
        f"{label}_DivMom": (1+ts_div).cumprod(),
        f"{label}_Graph": (1+ts_graph).cumprod(),
        "SP500": (1+ts_bench).cumprod()
    })
    return metrics, eq

# -------- Prepare four embeddings --------
embeddings = {
    "QLSTM_FreeRun": fr[["f1","f2"]],                 # Core seq2seq embedding
    "PCA2": None,                                      # Computed below
    "Random2D": pd.DataFrame(np.random.randn(len(tickers),2), index=tickers, columns=["f1","f2"]),
    "Shuffled2D": fr.sample(fr.shape[0]).set_index(fr.index)[["f1","f2"]]  # Shuffle ticker-to-embedding mapping
}
# PCA(2) on weekly returns (PCA applied to stocks × time matrix)
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore", message="n_init", category=FutureWarning)
pca = PCA(n_components=2).fit(ret.fillna(0).T.values)
pca_scores = pca.transform(ret.fillna(0).T.values)
embeddings["PCA2"] = pd.DataFrame(pca_scores, index=tickers, columns=["f1","f2"])

# -------- Run backtest for each embedding (same parameters) --------
all_metrics = {}
all_eq = []
for name, emb in embeddings.items():
    m, eq = run_backtest_with_embedding(emb, label=name)
    all_metrics[name] = m
    eq = eq.drop(columns=["SP500"], errors="ignore")
    all_eq.append(eq)


# Output performance comparison table
table = pd.concat({k: v for k, v in all_metrics.items()}, axis=1).round(4)
table.to_csv(os.path.join(OUT, "attribution_metrics.csv"))

# Interactive equity curve chart (four embeddings)
eq_all = pd.concat(all_eq, axis=1).dropna(how="all")

sp = (1 + gspc.loc[eq_all.index]).cumprod()
eq_all["SP500"] = sp.values

fig_eq = go.Figure()
for col in eq_all.columns:
    fig_eq.add_trace(go.Scatter(x=eq_all.index, y=eq_all[col], mode="lines", name=col))
fig_eq.update_layout(title="Equity Curves across embeddings",
                     xaxis_title="Date", yaxis_title="Net Value")
fig_eq.write_html(os.path.join(OUT, "equity_curves_across_embeddings.html"))

# Interactive table (performance comparison)
def table_html(df: pd.DataFrame, title: str, path: str):
    # Flatten MultiIndex column names to avoid frontend library incompatibility
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" | ".join(map(str, c)) for c in df.columns]
    fig = go.Figure(data=[go.Table(
        header=dict(values=["<b>"+c+"</b>" for c in df.columns],
                    fill_color="#1f2937", font_color="white", align="center"),
        cells=dict(values=[df[c] for c in df.columns], align="center")
    )])
    fig.update_layout(title=title)
    fig.write_html(path)

table_html(table, "Attribution metrics (excess over RF)", os.path.join(OUT, "attribution_metrics.html"))

# -------------------- Correlation: Representation Quality vs Strategy Performance (cross-embedding) --------------------
# Use mapping_avg_distance (TF vs FR) as quality proxy (only QLSTM has this metric)
qual = pd.Series({"QLSTM_FreeRun": map_gap, "PCA2": np.nan, "Random2D": np.nan, "Shuffled2D": np.nan}, name="TF_FR_mapping_gap")
cagr_vs_gap = pd.Series({k: all_metrics[k].loc["CAGR","RBF-Graph (excess)"] for k in all_metrics}, name="CAGR_graph")
corr_spear = qual.corr(cagr_vs_gap, method="spearman")
pd.Series({"spearman_gap_vs_CAGR_graph": corr_spear}).to_csv(os.path.join(OUT, "quality_vs_perf.csv"))

# ==================== 4) Sector-based: Cross-embedding Validation (NMI / ARI / Distance Gap) ====================
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI

def _norm_ticker_case(x: str) -> str:
    return str(x).strip().upper()

def _sector_df(path: str) -> pd.DataFrame:
    sec = pd.read_csv(path)
    sec.columns = [c.strip().lower() for c in sec.columns]
    if not {"ticker","sector"}.issubset(sec.columns):
        raise ValueError("ticker_to_sector.csv must contain at least the columns: ticker, sector")
    sec["ticker"] = sec["ticker"].map(_norm_ticker_case)
    return sec.set_index("ticker")

def _sector_metrics_for_embedding(embedding_2d: pd.DataFrame,
                                  sector_csv: str,
                                  out_dir: str,
                                  label: str):
    """
    Returns: dict(gap, avg_same, avg_diff, NMI, ARI, n_used, n_sectors)
    Also outputs individual files (metrics CSV + scatter plot + heatmap),
    with automatic handling of NaN values and title formatting.
    """
    os.makedirs(out_dir, exist_ok=True)
    if sector_csv is None or not os.path.exists(sector_csv):
        print(f"[Sector-{label}] ticker_to_sector.csv not found, skipping.")
        return {"gap": np.nan, "avg_same": np.nan, "avg_diff": np.nan,
                "NMI": np.nan, "ARI": np.nan, "n_used": 0, "n_sectors": 0}

    # Read and normalize sector data
    sec = pd.read_csv(sector_csv)
    sec.columns = [c.strip().lower() for c in sec.columns]
    if not {"ticker","sector"}.issubset(sec.columns):
        raise ValueError("ticker_to_sector.csv must contain at least the columns: ticker, sector")
    sec["ticker"] = sec["ticker"].astype(str).str.strip().str.upper()
    sec["sector"] = sec["sector"].astype(str).str.strip()
    sec = sec.set_index("ticker")

    # Align to embedding
    Z = embedding_2d.copy()
    Z.index = Z.index.astype(str).str.strip().str.upper()
    common = Z.index.intersection(sec.index)

    # Keep only aligned rows and ensure float dtype
    Z = Z.loc[common, ["f1","f2"]].astype(float)
    lab = sec.loc[common, "sector"]

    # Filter out invalid rows (NaN or empty sector label)
    mask_valid = (
        Z[["f1","f2"]].notna().all(axis=1) &
        lab.notna() &
        (lab.astype(str).str.len() > 0)
    )
    Z = Z.loc[mask_valid]
    lab = lab.loc[mask_valid]

    if Z.shape[0] < 2:
        print(f"[Sector-{label}] Insufficient valid samples ({Z.shape[0]}), skipping.")
        pd.Series({
            "avg_distance_same_sector": np.nan,
            "avg_distance_diff_sector": np.nan,
            "distance_gap(diff - same)": np.nan,
            "num_tickers_used": int(Z.shape[0]),
        }).to_csv(os.path.join(out_dir, f"sector_structure_metrics_{label}.csv"))
        pd.Series({
            "NMI(latent_kmeans_vs_sector)": np.nan,
            "ARI(latent_kmeans_vs_sector)": np.nan,
            "num_sectors": 0
        }).to_csv(os.path.join(out_dir, f"sector_alignment_metrics_{label}.csv"))
        return {"gap": np.nan, "avg_same": np.nan, "avg_diff": np.nan,
                "NMI": np.nan, "ARI": np.nan, "n_used": int(Z.shape[0]), "n_sectors": 0}

    # Distance matrix (Euclidean in latent space)
    D = squareform(pdist(Z.values, metric="euclidean"))
    sectors = lab.values

    # Structural statistics: same-sector vs different-sector distances
    same_mask = sectors[:, None] == sectors[None, :]
    np.fill_diagonal(same_mask, False)
    diff_mask = ~same_mask

    same_vals = D[same_mask]
    diff_vals = D[diff_mask]
    avg_same = float(np.nanmean(same_vals)) if same_vals.size else np.nan
    avg_diff = float(np.nanmean(diff_vals)) if diff_vals.size else np.nan
    gap = (avg_diff - avg_same) if (same_vals.size and diff_vals.size) else np.nan

    pd.Series({
        "avg_distance_same_sector": avg_same,
        "avg_distance_diff_sector": avg_diff,
        "distance_gap(diff - same)": gap,
        "num_tickers_used": int(len(Z))
    }).to_csv(os.path.join(out_dir, f"sector_structure_metrics_{label}.csv"))

    # KMeans alignment (k = number of sectors); skip if insufficient data
    uniq = pd.unique(lab)
    k = len(uniq)
    nmi = ari = np.nan
    if k >= 2 and len(Z) >= k:
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            pred = km.fit_predict(Z.values)
            nmi = float(NMI(lab.values, pred))
            ari = float(ARI(lab.values, pred))
        except Exception as e:
            print(f"[Sector-{label}] KMeans failed (skipped): {e}")

    pd.Series({
        "NMI(latent_kmeans_vs_sector)": nmi,
        "ARI(latent_kmeans_vs_sector)": ari,
        "num_sectors": int(k)
    }).to_csv(os.path.join(out_dir, f"sector_alignment_metrics_{label}.csv"))

    # --------- Interactive visualizations ---------
    def fmt(x):
        try:
            return f"{float(x):.3f}"
        except Exception:
            return "nan"

    title_suffix = f"(gap={fmt(gap)}, NMI={fmt(nmi)}, ARI={fmt(ari)})"

    df_plot = Z.copy()
    df_plot["sector"] = lab.values
    fig_scatter = px.scatter(df_plot, x="f1", y="f2", color="sector",
                             hover_name=df_plot.index,
                             title=f"{label}: latent by sector {title_suffix}")
    fig_scatter.write_html(os.path.join(out_dir, f"latent_by_sector_{label}.html"))

    order = np.argsort(lab.values)
    D_sorted = D[order][:, order]
    tick_sorted = Z.index.to_numpy()[order]
    fig_heat = px.imshow(pd.DataFrame(D_sorted, index=tick_sorted, columns=tick_sorted),
                         aspect="auto", color_continuous_scale="Viridis",
                         title=f"{label}: latent distance heatmap (sorted by sector)")
    fig_heat.update_xaxes(side="top", tickangle=45)
    fig_heat.write_html(os.path.join(out_dir, f"sector_distance_heatmap_{label}.html"))

    return {"gap": gap, "avg_same": avg_same, "avg_diff": avg_diff,
            "NMI": nmi, "ARI": ari, "n_used": int(len(Z)), "n_sectors": int(k)}

def evaluate_sector_across_embeddings(embeddings: dict,
                                      sector_csv: str,
                                      out_dir: str):
    """
    embeddings: dict[name -> DataFrame(index=ticker, cols=['f1','f2'])]
    Outputs sector_comparison_metrics.csv and a corresponding bar chart HTML.
    """
    rows = []
    for name, emb in embeddings.items():
        print(f"[Sector] evaluating {name} ...")
        m = _sector_metrics_for_embedding(emb, sector_csv, out_dir, label=name)
        m["embedding"] = name
        rows.append(m)
    comp = pd.DataFrame(rows).set_index("embedding")[["gap", "NMI", "ARI", "avg_same", "avg_diff", "n_used", "n_sectors"]]
    comp.to_csv(os.path.join(out_dir, "sector_comparison_metrics.csv"))

    # Visualization: three key metrics — NMI / ARI / gap
    melt = comp.reset_index().melt(id_vars="embedding", value_vars=["NMI", "ARI", "gap"],
                                   var_name="metric", value_name="value")
    fig = px.bar(melt, x="embedding", y="value", color="metric", barmode="group",
                 title="Sector alignment comparison across embeddings (NMI / ARI / Gap)")
    fig.write_html(os.path.join(out_dir, "sector_comparison_bars.html"))
    print("[Sector] Summary saved:",
          os.path.join(out_dir, "sector_comparison_metrics.csv"),
          os.path.join(out_dir, "sector_comparison_bars.html"), sep="\n  ")

# Run sector validation (if ticker_to_sector.csv is available)
try:
    evaluate_sector_across_embeddings(embeddings, SECTOR_CSV, OUT)
except Exception as e:
    print(f"[Sector] Cross-embedding validation error: {e}")

print("== Done ==")
print(f"- Fidelity metrics: {os.path.join(OUT, 'fidelity_metrics.csv')}")
print(f"- Structure metrics: {os.path.join(OUT, 'structure_metrics.csv')}")
print(f"- Attribution metrics table: {os.path.join(OUT, 'attribution_metrics.csv')}")
print(f"- Equity curves (all embeddings): {os.path.join(OUT, 'equity_curves_across_embeddings.html')}")
print(f"- Quality↔Performance corr: {os.path.join(OUT, 'quality_vs_perf.csv')}")
print(f"- Sector comparison: {os.path.join(OUT, 'sector_comparison_metrics.csv')} (and sector_comparison_bars.html)")