"""
count_usable_paths.py

This code counts how many unique (strike, expiry) option paths 
meet different thresholds for:
1. minimum path length (number of consecutive observations), and
2. maximum gap between observations (in calendar days).

Load datasets train, val, test with option contracts identified by strike price and expiry date
Break each contract's time series into contiguous runs, allowing for gaps up to a chosen number of days.
For each (min_len, days) combination:
1. Count how many usable paths exist overall and within each market regime.
2. Print results for train, validation, and test sets.

This is useful for preparing data for sequence models where 
long, continuous time series are required, and for determining the optimal min lengths and days for the 
preprocess_lstm_option_data.py file to follow. 
"""

import pandas as pd

#Parameters to test
MIN_LENGTHS = [20, 30, 40, 50]
DAYS_ALLOWED = [0, 1, 3, 5]

def count_usable_paths(csv_path: str, min_len: int, days: int):
    """Count usable paths in one dataset for given min length and max gap days"""
    df = pd.read_csv(csv_path, parse_dates=["date","exdate"])
    
    df["contract"] = (df["strike_price"].astype(str) + "_" +df["exdate"].dt.strftime("%Y-%m-%d"))
    df = df.sort_values(["contract", "date"])
    
    df["prev_date"] = df.groupby("contract")["date"].shift(1)
    df["is_next"] = (df["date"] - df["prev_date"]).dt.days <= days
    df["run_group"] = (~df["is_next"].fillna(False)).cumsum().astype(str) + "_" + df["contract"]
    segs = (df.groupby(["market_regime", "contract", "run_group"]).agg(length=("date", "count")).reset_index())
    
    usable = segs[segs["length"] >= min_len]
    counts_by_regime = usable.groupby("market_regime")["contract"].nunique()
    return usable, counts_by_regime

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        path = f"data_{split}.csv"
        print(f"\n{split.upper()} Data")
        for days in DAYS_ALLOWED:
            for min_len in MIN_LENGTHS:
                usable, counts = count_usable_paths(path, min_len, days)
                total = usable["contract"].nunique()
                print(f"Gap <= {days} days with Min length >= {min_len:2d} gives {total} usable contracts")
                for regime, c in counts.items():
                    print(f"{regime:10s}: {c} contracts")
