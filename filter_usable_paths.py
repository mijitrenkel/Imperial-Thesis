"""
This code filters option path_id time series from raw CSV files to retain only 
usable paths of sufficient length, intended for LSTM training

1.Load Data: Reads CSV files of train, val and test
2.path_id Identification: Creates a unique path_id ID by combining strike price and expiration date,
then sorts by path_id and date
3.Continuity Detection:Assigns previous date for the same path_id, marks is_next True if the current date 
is within 3 days (allows for weekends + holidays) of the previous 
4.Run Segmentation: Splits each path_id's history into contiguous segments, breaking whenever gaps exceed 3 days
5.Length Filtering: Retains only those segments with at least min_len observations of 30
6.Output: Saves the filtered train, validation, and test
"""
import pandas as pd
MIN_LENGTH = 30

def filter_usable_paths(csv_path: str, min_len: int = MIN_LENGTH):
    df = pd.read_csv(csv_path, parse_dates=["date","exdate"])
    df["path_id"] = df["strike_price"].astype(str) + "_" + df["exdate"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["path_id", "date"])
    df["prev_date"] = df.groupby("path_id")["date"].shift(1)
    df["is_next"] = (df["date"] - df["prev_date"]).dt.days <= 3
    df["run_group"] = (~df["is_next"].fillna(False)).cumsum().astype(str) + "_" + df["path_id"]
    
    segs = (df.groupby(["path_id", "run_group"]).agg(length=("date", "count")).reset_index())
    
    usable_segs = segs[segs["length"] >= min_len]
    mask = df.set_index(["path_id", "run_group"]).index.isin(
        usable_segs.set_index(["path_id", "run_group"]).index)
    df_usable = df[mask].copy()
    
    return df_usable

df_train = filter_usable_paths("data_train.csv", MIN_LENGTH)
df_val = filter_usable_paths("data_val.csv", MIN_LENGTH)
df_test = filter_usable_paths("data_test.csv", MIN_LENGTH)

df_train.to_csv("df_train.csv", index=False)
df_val.to_csv("df_val.csv", index=False)
df_test.to_csv("df_test.csv", index=False)
