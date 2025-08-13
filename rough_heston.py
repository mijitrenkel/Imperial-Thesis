"""
Generate Rough-Heston paths
"""
from __future__ import annotations
import json, math, functools
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy.fft import rfft, irfft
try:
    from scipy.fft import next_fast_len 
except ModuleNotFoundError:
    def next_fast_len(n:int)->int:        
        return 1 << (n-1).bit_length()

JSON_IN, OUT_NORMAL, OUT_CRISIS = "rv_params.json", "rough_paths_normal.csv", "rough_paths_crisis.csv"
N_NORMAL, N_CRISIS = 2000, 500
TENOR_DAYS = [30,60,90,180,360]
MC_SIMS = 5000
SEED = 2
master_rng = np.random.default_rng(SEED)

def moneyness_bucket(x:float)->str|float:
    if math.isnan(x): return np.nan
    return (
        "Deep ITM" if x> .20 else
        "ITM" if x> .05 else
        "ATM" if x>=-.05 else
        "OTM" if x>=-.20 else
        "Deep OTM"
    )

def fractional_kernel_weights(H: float, N: int, dt: float) -> np.ndarray:
    """Discrete rough-Heston weights
      w_j = [ (j+1)^(H+1/2) - j^(H+1/2) ] / (H+1/2) * dt^(H-1/2)
    for j=0..N-1.  This is finite at j=0.
    """
    j = np.arange(N, dtype=float)
    factor = dt ** (H - 0.5) / (H + 0.5)
    return ((j + 1) ** (H + 0.5) - j ** (H + 0.5)) * factor

def make_hybrid_kernel(H: float, N: int, dt: float, m0: int):
    g = fractional_kernel_weights(H, N, dt)
    return g[:m0], g

@functools.lru_cache(maxsize=None)
def cholesky_matrix(g_chol:tuple[float],dt:float)->np.ndarray:
    m0 = len(g_chol)
    C = np.empty((m0,m0))
    for i in range(m0):
        for j in range(i+1):
            C[i,j] = C[j,i] = sum(g_chol[k]*g_chol[k+abs(i-j)] for k in range(m0-abs(i-j)))
    return np.linalg.cholesky(C+1e-18)                      

def fft_convolve_paths(incr:np.ndarray, g:np.ndarray)->np.ndarray:
    M,N = incr.shape
    L = next_fast_len(2*N-1)
    Gf = rfft(np.pad(g,(0,L-N)))
    Ff = rfft(np.pad(incr,((0,0),(0,L-N))),axis=1)
    out = irfft(Ff*Gf,axis=1)[:,:N]
    return out

def simulate_rough_heston(
        S0:float,V0:float,r:float,kappa:float,theta:float,
        xi:float,rho:float,H:float,T:float,N:int,M:int,
        rng:np.random.Generator,m0:int=20
):
    dt = T/N
    Z1,Z2 = rng.normal(size=(2,M,N))
    dW1 = np.sqrt(dt)*Z1
    dW2 = rho*dW1 + np.sqrt(1-rho**2)*np.sqrt(dt)*Z2

    g_chol, g_full = make_hybrid_kernel(H,N,dt,m0)
    Lchol = cholesky_matrix(tuple(g_chol),dt)       
    J = np.zeros((M,N))
    if m0>0:
        chol_incr = dW2[:,:m0]          
        J[:,:m0]= (chol_incr @ Lchol.T)

    J_fft = fft_convolve_paths(dW2,g_full)
    J[:,m0:] = J_fft[:,m0:]

    logS = np.full(M, math.log(S0))
    S = np.empty((M,N+1)); S[:,0]=S0
    V = np.empty((M,N+1)); V[:,0]=V0

    for n in range(N):
        drift = kappa*(theta - V[:,n])*dt
        V[:,n+1] = np.maximum(V[:,n] + drift + xi*J[:,n], 1e-10)
        logS += (r - 0.5*V[:,n])*dt + np.sqrt(V[:,n])*dW1[:,n]
        S[:,n+1]= np.exp(logS)

    return S,V

def worker(params):
    S0 = float(params["S0"])
    r = float(params["r"])
    H = float(params["H"])
    kappa = float(params["kappa"])
    theta = float(params["theta"])
    xi = float(params["xi"])
    rho = float(params["rho"])
    regime = params["regime"]
    strike_frac = float(params["strike_frac"])
    path_id = params["id"]
    V0 = float(params.get("V0", theta))

    rng = np.random.default_rng(master_rng.integers(0,2**32-1))
    tenor = rng.choice(TENOR_DAYS)       
    T = tenor / 252.0
    rows = []

    S_paths, V_paths = simulate_rough_heston(S0, V0, r, kappa, theta, xi, rho, H,T, tenor, MC_SIMS, rng, m0=20)
    S_T = S_paths[:,-1]
    discount= math.exp(-r*T)
    K = strike_frac * S0

    for t in range(tenor + 1):
        raw_S = S_paths[0, t]
        raw_V = V_paths[0, t]
        S_t = S_paths[:, t]
        tau_remain = tenor - t
        discount= math.exp(-r*(tau_remain/252))
        pathwise = (S_T>K)*(S_T/S_t)
        delta_t = discount*np.mean(pathwise)
        delta = float(np.clip(delta_t,0.0,1.0))
        price_t  = discount * np.maximum(S_T - K, 0).mean()
        price = float(np.clip(price_t,0.0,None))
        lm  = math.log(raw_S / K)

        rows.append({
            "regime": regime,
            "path_id": path_id,
            "tenor": tenor,
            "step": t,
            "tau_remain": tau_remain,
            "S": raw_S,
            "V": raw_V,
            "strike_frac": strike_frac,
            "r": r,
            "delta": delta,
            "price": price,
            "log_moneyness": lm,
            "moneyness_category": moneyness_bucket(lm),
        })

    return rows
def main():
    draws = json.loads(Path(JSON_IN).read_text())
    normal = [d for d in draws if d["regime"]=="normal"][:N_NORMAL]
    crisis = [d for d in draws if d["regime"]=="crisis"][:N_CRISIS]
    for i,d in enumerate(normal): d["id"]=f"normal_{i:04d}"
    for i,d in enumerate(crisis): d["id"]=f"crisis_{i:04d}"

    for tag,sample,out_csv in [("normal",normal,OUT_NORMAL),
                               ("crisis",crisis,OUT_CRISIS)]:
        print(f"{tag}: {len(sample)} param draws . tenors={TENOR_DAYS}")
        with Pool(cpu_count()) as pool:
            rows = [r for sub in pool.map(worker,sample) for r in sub]
        pd.DataFrame(rows).to_csv(out_csv,index=False)
        print(f" {len(rows):,} rows -> {out_csv}")

if __name__=="__main__":
    main()

    for i, d in enumerate(normals):
        d["id"] = f"normal_{i:04d}"
    for i, d in enumerate(crises):
        d["id"] = f"crisis_{i:04d}"

    samp_norm = master_rng.choice(normals, size=min(N_NORMAL, len(normals)), replace=False)
    samp_crisis = master_rng.choice(crises, size=min(N_CRISIS, len(crises)), replace=False)

    for regime, sample, out_csv in [
        ("normal", samp_norm, OUT_NORMAL),
        ("crisis", samp_crisis, OUT_CRISIS),
    ]:
        print(f"\n{regime}: {len(sample)} paths, tenors={TENOR_DAYS}")
        with Pool(cpu_count()) as pool:
            all_rows = pool.map(worker, sample)
        df = pd.DataFrame([row for sublist in all_rows for row in sublist])
        df.to_csv(out_csv, index=False)
        print(f"{len(df):,} rows to {out_csv}")

if __name__ == "__main__":
    main()
