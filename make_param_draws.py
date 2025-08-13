"""
make_param_draws.py
Generates random parameter draws for Black-Scholes and Rough Vol Model
under two market regimes: 'normal' and 'crisis'.
Draws are constrained by empirical bounds and theoretical requirement of the Feller condition.
Outputs are saved to JSON for later simulation use.
"""
import json
import math
import random
from typing import Dict, Tuple
import numpy as np

SEED = 2
random.seed(SEED)
np.random.seed(SEED)
N_NORM   = 5000
N_CRISIS = 2000

#Buffers for empirical bounds
BUFFER_NORMAL        = 0.05
BUFFER_CRISIS        = 0.10
ASYM_BUFFER_NORMAL_U = 0.05
ASYM_BUFFER_CRISIS_U = 0.10
ASYM_BUFFER_NORMAL_D = 0.0025
ASYM_BUFFER_CRISIS_D = 0.05

# Strike grids
STRIKE_GRID_NORMAL = np.linspace(0.92, 1.20, 15)
STRIKE_GRID_CRISIS = np.linspace(0.83, 1.50, 15)

# Spot range
S0_MIN, S0_MAX =  752.44 * 0.90,  2104.18 * 1.10

# Helper sampling functions
def log_uniform(lo: float, hi: float) -> float:
    return math.exp(random.uniform(math.log(lo), math.log(hi)))

def uniform(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)

def sample_one(config: Dict[str, Tuple[float, float, bool]]) -> Dict[str, float]:
    out = {
        k: (log_uniform(lo, hi) if is_log else uniform(lo, hi))
        for k, (lo, hi, is_log) in config.items()
    }
    if "rho" in out:
        out["rho"] = max(min(out["rho"], 1.0), -1.0)
    return out

def ensure_feller(p: Dict[str, float]) -> bool:
    return 2 * p["kappa"] * p["theta"] > p["xi"] ** 2

def expand_block(raw: Dict[str, float], regime: str):
    strikes = STRIKE_GRID_NORMAL if regime == "normal" else STRIKE_GRID_CRISIS
    for kf in strikes:
        yield {**raw, "regime": regime, "strike_frac": kf}

def buffered(lo: float, hi: float, buffer: float) -> Tuple[float, float]:
    return lo * (1 - buffer), hi * (1 + buffer)

def asym_buffered(lo: float, hi: float, buff_lo: float, buff_hi: float) -> Tuple[float, float]:
    return lo * (1 - buff_lo), hi * (1 + buff_hi)

bs_PRIOR = {
    "normal": {
        "r":     (*buffered(0.00159, 0.05053, BUFFER_NORMAL), False),
        "sigma": (*buffered(0.10434, 0.39677, BUFFER_NORMAL), True),
    },
    "crisis": {
        "r":     (*buffered(0.00868, 0.04888, BUFFER_CRISIS), False),
        "sigma": (*buffered(0.15963, 0.63448, BUFFER_CRISIS), True),
    },
}

rv_PRIOR = {
    "normal": {
        "r":     (*bs_PRIOR["normal"]["r"][:2], False),
        "H":     (*buffered(0.08, 0.12, BUFFER_NORMAL), False),
        "kappa": (*buffered(0.8, 4.8, BUFFER_NORMAL), False),
        "theta": (*buffered(0.016, 0.048, BUFFER_NORMAL), False),
        "rho":   (*asym_buffered(-0.8, -0.5, ASYM_BUFFER_NORMAL_D, ASYM_BUFFER_NORMAL_U), False),
        "xi":    (*asym_buffered(0.3, 0.5, ASYM_BUFFER_NORMAL_D, ASYM_BUFFER_NORMAL_U), True),
    },
    "crisis": {
        "r":     (*bs_PRIOR["crisis"]["r"][:2], False),
        "H":     (*buffered(0.07, 0.13, BUFFER_CRISIS), False),
        "kappa": (*buffered(4.0, 9.6, BUFFER_CRISIS), False),
        "theta": (*buffered(0.024, 0.06, BUFFER_CRISIS), False),
        "rho":   (*asym_buffered(-0.82, -0.57, ASYM_BUFFER_CRISIS_D, ASYM_BUFFER_CRISIS_U), False),
        "xi":    (*asym_buffered(0.45, 0.58, ASYM_BUFFER_CRISIS_D, ASYM_BUFFER_CRISIS_U), True),
    },
}

def draw_bs(regime: str, n_draws: int):
    cfg = bs_PRIOR[regime]
    needed = n_draws * len(STRIKE_GRID_NORMAL if regime == "normal" else STRIKE_GRID_CRISIS)
    acc = []
    while len(acc) < needed:
        raw = sample_one(cfg)
        raw["S0"] = uniform(S0_MIN, S0_MAX)
        acc.extend(expand_block(raw, regime))
    return acc[:needed]

def draw_rv(regime: str, n_draws: int):
    cfg = rv_PRIOR[regime]
    needed = n_draws * len(STRIKE_GRID_NORMAL if regime == "normal" else STRIKE_GRID_CRISIS)
    acc = []
    while len(acc) < needed:
        raw = sample_one(cfg)
        raw["S0"] = uniform(S0_MIN, S0_MAX)
        if not ensure_feller(raw):
            continue
        acc.extend(expand_block(raw, regime))
    return acc[:needed]

if __name__ == "__main__":
    bs_params  = draw_bs("normal",  N_NORM) + draw_bs("crisis",  N_CRISIS)
    rv_params = draw_rv("normal", N_NORM) + draw_rv("crisis", N_CRISIS)

    with open("bs_params.json",  "w") as f:
        json.dump(bs_params,  f, indent=2)
    with open("rv_params.json", "w") as f:
        json.dump(rv_params, f, indent=2)

    print(f"Saved bs_params.json  with {len(bs_params):,} rows")
    print(f"Saved rv_params.json with {len(rv_params):,} rows")
