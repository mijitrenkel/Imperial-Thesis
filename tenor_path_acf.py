import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import random

FILES = {
    "RV Normal": "rough_paths_normal.csv",
    "RV Crisis": "rough_paths_crisis.csv",
    "BS Normal" : "bs_paths_normal.csv",
    "BS Crisis": "bs_paths_crisis.csv"
}
MAX_TENORS = 5
NLAGS = 40
RANDOM_SEED = 2

def sample_one_path_per_tenor(df: pd.DataFrame, max_tenors: int = 5):
    tenors = [t for t in pd.Series(df["tenor"]).dropna().unique().tolist()]
    if len(tenors) > max_tenors:
        tenors = random.sample(tenors, max_tenors)

    selections = []
    for t in tenors:
        ids = pd.Series(df.loc[df["tenor"] == t, "path_id"]).dropna().unique().tolist()
        if not ids:
            continue
        pid = random.choice(ids)
        selections.append((t, pid))
    return selections

def safe_acf(series: np.ndarray, lags: int):
    if len(series) < 2:
        return None
    nlags = max(1, min(lags, len(series) - 1))
    return acf(series, nlags=nlags, fft=True)

def plot_s_overlay(label: str, df: pd.DataFrame, selections):
    plt.figure(figsize=(12, 6))
    for tenor, pid in selections:
        sub = df[(df["tenor"] == tenor) & (df["path_id"] == pid)].sort_values("step")
        if sub.empty:
            continue
        plt.plot(sub["step"].values, sub["S"].values, alpha=0.9, label=f"tenor={tenor}, path={pid}")
    plt.title(f"{label} S vs. Steps (1 random path per tenor)")
    plt.xlabel("Step")
    plt.ylabel("Spot Price (S)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_acf_overlays(label: str, df: pd.DataFrame, selections, kind: str, lags: int):
    plt.figure(figsize=(12, 6))
    for tenor, pid in selections:
        sub = df[(df["tenor"] == tenor) & (df["path_id"] == pid)].sort_values("step")
        if sub.empty:
            continue

        if kind == "returns":
            prices = sub["S"].values
            series = np.diff(np.log(prices)) if len(prices) >= 2 else np.array([])
            title = "ACF of Log-Returns"
            ylabel = "ACF"
        elif kind == "vol":
            series = sub["inst_vol"].values
            title = "ACF of Volatility (inst_vol)"
            ylabel = "ACF"
        else:
            continue

        acf_vals = safe_acf(series, lags)
        if acf_vals is None:
            continue

        x = np.arange(len(acf_vals))
        plt.plot(x, acf_vals, marker='o', linewidth=1.5, label=f"tenor={tenor}, path={pid}")

    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"{label} — {title} (1 random path per tenor)")
    plt.xlabel("Lag")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    for label, fname in FILES.items():
        df = pd.read_csv(fname)
        print(f"\nProcessing {label}-{fname}")
        selections = sample_one_path_per_tenor(df, max_tenors=MAX_TENORS)
        plot_s_overlay(label, df, selections)
        plot_acf_overlays(label, df, selections, kind="returns", lags=NLAGS)
        plot_acf_overlays(label, df, selections, kind="vol", lags=NLAGS)

if __name__ == "__main__":
    main()
