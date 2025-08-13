import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error, mean_absolute_error

EPS = 1e-8
COLS = dict(
    path_id="path_id",
    step="step",
    S="S",
    opt_mid="price",
    opt_bid="best_bid",
    opt_ask="best_offer",
    r="r",
    inst_vol="inst_vol",
    tau_remain="tau_remain",
    strike_frac="strike_frac",
    delta_true="delta",
    delta_pred="delta_pred",
)

TAU_BINS   = [0, 30, 90, 180, 360, np.inf]
TAU_LABELS = ["<=30d","31-90d","91-180d","181-360d",">360d"]
M_BINS     = [-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf]
M_LABELS   = ["deep_OTM","OTM","ATM","ITM","deep_ITM"]

COST_MULTS = [0.0, 0.5, 1.0]
RH_FREQS   = [1, 2, 5]

csv_dict = {
    "out/bs_comb_synth_ft/synth+finetune_bs_comb_delta_pred.csv":        "bs_comb_delta_pred",
    "out/rough_comb_synth_ft/synth+finetune_rough_comb_delta_pred.csv":  "rough_comb_delta_pred",
    "out/rough_crisis_synth_ft/synth+finetune_rough_crisis_delta_pred.csv":"rough_crisis_delta_pred",
    "out/bs_crisis_synth_ft/synth+finetune_bs_crisis_delta_pred.csv":    "bs_crisis_delta_pred",
    "out/rough_synth_ft/synth+finetune_rough_norm_delta_pred.csv":       "rough_norm_delta_pred",
    "out/bs_synth_ft/synth+finetune_bs_norm_delta_pred.csv":             "bs_norm_delta_pred",
    "out/real/real-only_real_delta_pred.csv":                            "real_delta_pred",
}

def _to_valid_array(values):
    a = np.asarray(values, dtype=float)
    return a[np.isfinite(a)]

def safe_mean(values):
    a = _to_valid_array(values)
    return float(a.mean()) if a.size else np.nan

def safe_var(values):
    a = _to_valid_array(values)
    return float(a.var()) if a.size >= 2 else np.nan

def es_left(pnl, alpha=0.05):
    x = np.asarray(pnl)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    k = max(1, int(np.ceil(alpha * x.size)))
    return float(np.mean(np.sort(x)[:k]))

def compute_delta_errors(y_true, y_pred):
    if len(y_true) == 0:
        return {"delta_MSE": np.nan, "delta_MAE": np.nan}
    return {
        "delta_MSE": mean_squared_error(y_true, y_pred),
        "delta_MAE": mean_absolute_error(y_true, y_pred),
    }
def pnl_exact_per_path(df, use_pred_delta=True):
    need = {COLS["opt_bid"], COLS["opt_ask"], COLS["path_id"], COLS["step"], COLS["S"]}
    if not need.issubset(df.columns):
        return {pid: np.nan for pid in df.get(COLS["path_id"], pd.Series(dtype=object)).unique()}
    out = {}
    for pid, grp in df.groupby(COLS["path_id"]):
        grp = grp.sort_values(COLS["step"])
        deltas = grp[COLS["delta_pred"]].to_numpy(float) if use_pred_delta else grp[COLS["delta_true"]].to_numpy(float)
        S  = grp[COLS["S"]].to_numpy(float)
        bb = grp[COLS["opt_bid"]].to_numpy(float)
        bo = grp[COLS["opt_ask"]].to_numpy(float)
        if len(grp) < 2:
            out[pid] = np.nan
            continue
        pnl, pos = 0.0, 0.0
        for t in range(1, len(grp)):
            desired = deltas[t-1]
            trade = desired - pos
            cost = trade * (bo[t] if trade > 0 else bb[t])  
            pnl += pos * (S[t] - S[t-1]) - cost            
            pos = desired
        out[pid] = pnl
    return out

def pnl_halfspread_per_path(df, use_pred_delta=True, cost_mult=1.0, k=1):
    need = {COLS["opt_bid"], COLS["opt_ask"], COLS["path_id"], COLS["step"], COLS["S"]}
    if not need.issubset(df.columns):
        return {pid: np.nan for pid in df.get(COLS["path_id"], pd.Series(dtype=object)).unique()}
    out = {}
    for pid, grp in df.groupby(COLS["path_id"]):
        grp = grp.sort_values(COLS["step"])
        deltas = grp[COLS["delta_pred"]].to_numpy(float) if use_pred_delta else grp[COLS["delta_true"]].to_numpy(float)
        S  = grp[COLS["S"]].to_numpy(float)
        bb = grp[COLS["opt_bid"]].to_numpy(float)
        bo = grp[COLS["opt_ask"]].to_numpy(float)
        if len(grp) < 2:
            out[pid] = np.nan
            continue
        pnl, pos = 0.0, 0.0
        for t in range(1, len(grp)):
            if (t-1) % k == 0:
                desired = deltas[t-1]
                trade = desired - pos
                half_spread = 0.5 * (bo[t] - bb[t])
                pnl -= abs(trade) * half_spread * cost_mult
                pos = desired
            pnl += pos * (S[t] - S[t-1])
        out[pid] = pnl
    return out

delta_rows, bucket_rows, sweep_rows = [], [], []
perpath_pnl = {}

for csv_path, pred_col in csv_dict.items():
    path = Path(csv_path)
    model_name = pred_col.replace("_delta_pred","")
    if not path.exists():
        print(f"[WARN] Missing: {csv_path}")
        continue

    df = pd.read_csv(path)
    if pred_col in df.columns and pred_col != COLS["delta_pred"]:
        df = df.rename(columns={pred_col: COLS["delta_pred"]})

    df["K"] = df["strike_price"]/1000 if "strike_price" in df.columns else np.nan
    core = df.dropna(subset=[COLS["delta_true"], COLS["delta_pred"]])
    derr = compute_delta_errors(core[COLS["delta_true"]], core[COLS["delta_pred"]])
    perpath_pnl[model_name] = pnl_exact_per_path(df, use_pred_delta=True)

    regime_iter = df.groupby("market_regime") if "market_regime" in df.columns else [("all", df)]
    for regime, grp in regime_iter:
        pnl_pred_map = pnl_exact_per_path(grp, True)
        pnl_true_map = pnl_exact_per_path(grp, False)
        pnl_pred = np.array(list(pnl_pred_map.values()), dtype=float)
        pnl_true = np.array(list(pnl_true_map.values()), dtype=float)

        delta_rows.append({
            "model": model_name,
            "regime" : regime,
            "delta_MSE" : derr["delta_MSE"],
            "delta_MAE" : derr["delta_MAE"],
            "mean_pnl_pred": float(np.nanmean(pnl_pred)),
            "mean_pnl_true": float(np.nanmean(pnl_true)),
            "var_pnl_pred" : float(np.nanvar(pnl_pred)),
            "var_pnl_true" : float(np.nanvar(pnl_true)),
            "es5_pred" : es_left(pnl_pred, 0.05),
            "es5_true" : es_left(pnl_true, 0.05),
            "es1_pred" : es_left(pnl_pred, 0.01),
            "es1_true" : es_left(pnl_true, 0.01),
            "n_paths" : int(len(pnl_pred)),
        })

        for cm, k in itertools.product(COST_MULTS, RH_FREQS):
            pnl_pred_h = np.array(list(pnl_halfspread_per_path(grp, True,  cost_mult=cm, k=k).values()), dtype=float)
            pnl_true_h = np.array(list(pnl_halfspread_per_path(grp, False, cost_mult=cm, k=k).values()), dtype=float)
            sweep_rows.append({
                "model" : model_name,
                "regime" : regime,
                "cost_mult" : cm,
                "freq_k" : k,
                "mean_pnl_pred": float(np.nanmean(pnl_pred_h)),
                "mean_pnl_true": float(np.nanmean(pnl_true_h)),
                "var_pnl_pred" : float(np.nanvar(pnl_pred_h)),
                "var_pnl_true" : float(np.nanvar(pnl_true_h)),
                "es5_pred" : es_left(pnl_pred_h, 0.05),
                "es5_true" : es_left(pnl_true_h, 0.05),
                "es1_pred" : es_left(pnl_pred_h, 0.01),
                "es1_true" : es_left(pnl_true_h, 0.01),
                "n_paths" : int(len(pnl_pred_h)),
            })

    # ---- Bucketed by moneyness × tau (exact P&L) ----
    if {COLS["S"], "K", COLS["tau_remain"]}.issubset(df.columns):
        df["m_bin"] = pd.cut(np.log(np.maximum(df[COLS["S"]],EPS)/np.maximum(df["K"],EPS)), bins=M_BINS, labels=M_LABELS)
        df["tau_bin"] = pd.cut(df[COLS["tau_remain"]], bins=TAU_BINS, labels=TAU_LABELS)
        group_cols = (["market_regime"] if "market_regime" in df.columns else []) + ["m_bin","tau_bin"]

        for keys, sub in df.groupby(group_cols, observed=True):
            if sub.empty:
                continue
            regime = keys[0] if "market_regime" in df.columns else "all"
            mbin = keys[1] if "market_regime" in df.columns else keys[0]
            tbin = keys[2] if "market_regime" in df.columns else keys[1]

            dd = sub.dropna(subset=[COLS["delta_true"], COLS["delta_pred"]])
            derr_b = compute_delta_errors(dd[COLS["delta_true"]], dd[COLS["delta_pred"]]) if not dd.empty else {"delta_MSE": np.nan, "delta_MAE": np.nan}

            pnl_pred_b = _to_valid_array(list(pnl_exact_per_path(sub, True).values()))
            pnl_true_b = _to_valid_array(list(pnl_exact_per_path(sub, False).values()))
            if pnl_pred_b.size == 0 and pnl_true_b.size == 0:
                continue


            bucket_rows.append({
                "model": model_name,
                "regime" : regime,
                "moneyness_bin": mbin,
                "tau_bin" : tbin,
                "delta_MSE" : derr_b["delta_MSE"],
                "delta_MAE" : derr_b["delta_MAE"],
                "mean_pnl_pred": float(np.nanmean(pnl_pred_b)),
                "mean_pnl_true": float(np.nanmean(pnl_true_b)),
                "var_pnl_pred" : float(np.nanvar(pnl_pred_b)),
                "var_pnl_true" : float(np.nanvar(pnl_true_b)),
                "es5_pred" : es_left(pnl_pred_b, 0.05),
                "es5_true" : es_left(pnl_true_b, 0.05),
                "es1_pred": es_left(pnl_pred_b, 0.01),
                "es1_true" : es_left(pnl_true_b, 0.01),
                "n_paths" : int(len(pnl_pred_b)),
            })

wilco_rows = []
models = list(perpath_pnl.keys())
for a, b in itertools.combinations(models, 2):
    pa, pb = set(perpath_pnl[a].keys()), set(perpath_pnl[b].keys())
    common = sorted(pa & pb)
    if not common:
        continue
    a_vals = np.array([perpath_pnl[a][pid] for pid in common], dtype=float)
    b_vals = np.array([perpath_pnl[b][pid] for pid in common], dtype=float)
    try:
        stat, pval = wilcoxon(a_vals, b_vals, zero_method="wilcox", alternative="greater")
    except ValueError:
        stat, pval = np.nan, np.nan
    wilco_rows.append({
        "model_A" : a,
        "model_B" : b,
        "n_paths" : int(len(common)),
        "A_mean" : float(np.nanmean(a_vals)),
        "B_mean" : float(np.nanmean(b_vals)),
        "A_es5" : es_left(a_vals, 0.05),
        "B_es5" : es_left(b_vals, 0.05),
        "A_es1" : es_left(a_vals, 0.01),
        "B_es1": es_left(b_vals, 0.01),
        "wilcoxon_stat": stat,
        "p_value(A>B)":  pval,
    })

pd.DataFrame(delta_rows).to_csv("eval_delta_overall.csv", index=False)
pd.DataFrame(bucket_rows).to_csv("eval_delta_buckets.csv", index=False)
pd.DataFrame(sweep_rows).to_csv("eval_cost_sweep.csv", index=False)
pd.DataFrame(wilco_rows).to_csv("eval_wilcoxon.csv", index=False)

print("eval_delta_overall.csv")
print("eval_delta_buckets.csv")
print("eval_cost_sweep.csv")
print("eval_wilcoxon.csv")
