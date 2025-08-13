"""
generate_bs_paths_by_regime.py

Simulate continuous BS hedging paths separately for normal and crisis regimes.
Outputs two CSVs: bs_paths_normal.csv, bs_paths_crisis.csv
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from numpy.random import default_rng


JSON_IN = "bs_params.json"
OUT_NORMAL = "bs_paths_normal.csv"
OUT_CRISIS = "bs_paths_crisis.csv"
N_NORMAL = 2000
N_CRISIS = 500
TENOR_DAYS = [30, 60, 90, 180, 360]
DT = 1.0/252.0
SEED = 2
master_rng = default_rng(SEED)

def call_price_and_delta(S, K, r, sigma, T):
    eps = 1e-8
    sqrtT = np.sqrt(np.maximum(T, eps))
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return price, delta

def simulate_gbm_path(rng, S0, r, sigma, n_steps, dt=DT):
    z = rng.normal(size=n_steps)
    dlogS = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z
    logS = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(dlogS)])
    return np.exp(logS)

def moneyness_bucket(x):
    if pd.isna(x):
        return np.nan
    if   x >  +0.20: return "Deep ITM"
    elif x > +0.05:  return "ITM"
    elif x >= -0.05: return "ATM"
    elif x >= -0.20: return "OTM"
    else:            return "Deep OTM"

def build_for_regime(draws, n_draws, regime_label):
    rng = default_rng(master_rng.integers(0,2**32-1))
    chosen = rng.choice(draws, size=min(n_draws, len(draws)), replace=False)
    rows = []

    for i, p in enumerate(tqdm(chosen, desc=f"{regime_label:>7}")):
        S0 = float(p["S0"])
        r = float(p["r"])
        sigma = float(p["sigma"])
        strike_frac = float(p["strike_frac"])
        K = strike_frac * S0
        path_id = f"{regime_label}_{i:04d}"
        
        tenor = int(rng.choice(TENOR_DAYS))
        S_path = simulate_gbm_path(rng,S0, r, sigma, tenor)
        taus = np.arange(tenor, -1, -1)
        Ts = taus * DT
        prices, deltas = call_price_and_delta(S_path, K, r, sigma, Ts)
        for step, (S_, tau_rem, pr, dl) in enumerate(zip(S_path, taus, prices, deltas)):
            log_moneyness = np.log(S_ / K)
            rows.append({
                "regime": regime_label,
                "path_id": path_id,
                "tenor": tenor,
                "step": step,
                "tau_remain": tau_rem,
                "S": S_,
                "strike_frac": strike_frac,
                "imp_vol": sigma,
                "r": r,
                "delta": dl,
                "price": pr,
                "log_moneyness": log_moneyness,
                "moneyness_category": moneyness_bucket(log_moneyness),
                })

    return pd.DataFrame(rows)

def main():
    with open(JSON_IN) as f:
        all_draws = json.load(f)

    normals = [p for p in all_draws if p["regime"] == "normal"]
    crises  = [p for p in all_draws if p["regime"] == "crisis"]

    df_norm = build_for_regime(normals, N_NORMAL, "normal")
    df_crisis = build_for_regime(crises,  N_CRISIS, "crisis")

    df_norm.to_csv(OUT_NORMAL,index=False)
    df_crisis.to_csv(OUT_CRISIS,index=False)
    print(f"{len(df_norm):,} rows to {OUT_NORMAL}")
    print(f"{len(df_crisis):,} rows to {OUT_CRISIS}")

if __name__ == "__main__":
    main()
