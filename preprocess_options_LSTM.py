"""
This code preprocesses the real filtered useable paths data sets for the LSTM 
Deduplication: For each (path_id, tau) pair, only the highest-volume observation is retained,
as having multiple rows wouldnt work in the LSTM. 

Feature Engineering: Constructs additional variables['tenor': total days between path_id's first trade and expiration,  
'step': sequential index within each path_id,'tau_remain': days remaining to expiration from trade date,  
'V': variance, defined as implied volatility squared, 'price': mid-quote between best bid and best offer.
The cleaned datasets are saved as new CSV files for modeling.
"""
import pandas as pd

def load_real_csv(path):
    df = pd.read_csv(path, dtype=str)
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    for col in ["best_bid","best_offer","volume","imp_vol","delta","S","tau","stripe_frac","r"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def dedup_by_path_id_tau(df):
    """Deduplicate so that only the highest-volume entry remains per (path_id, tau)."""
    df = df.copy()
    df["tau"] = df["tau"].astype(int)
    df = df.sort_values(["path_id","tau","volume"], ascending=[True,True,False])
    return df.drop_duplicates(subset=["path_id","tau"], keep="first").reset_index(drop=True)

def add_synthetic_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    first_dates = df.groupby("path_id")["date"].transform("min")
    df["tenor"] = (df["exdate"] - first_dates).dt.days
    df = df.sort_values(["path_id","date"])
    df["step"] = df.groupby("path_id").cumcount()
    df["tau_remain"] = (df["exdate"] - df["date"]).dt.days
    df["V"] = df["imp_vol"] ** 2
    df["inst_vol"] = df["imp_vol"]
    df["price"] = (df["best_bid"] + df["best_offer"]) / 2
    df["tau_norm"] = df["tau_remain"] / df["tenor"]
    return df

if __name__ == "__main__":
    df_train = load_real_csv("df_train.csv")
    df_val   = load_real_csv("df_val.csv")
    df_test  = load_real_csv("df_test.csv")

    #Preprocess
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        df = dedup_by_path_id_tau(df)
        df = add_synthetic_cols(df)
        df.to_csv(f"df_{name}_clean.csv", index=False)