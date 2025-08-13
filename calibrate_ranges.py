"""
calibrate_ranges.py

This code computes widened parameter ranges for variables: r, S, implied volatility, strike fraction, tau, log-moneyness
under different market regimes normal and crisis

For each regime:
    Compute the 5th and 95th percentiles of each variable
    Print the final calibrated intervals
"""
import pandas as pd
COLUMNS_TO_DESCRIBE = ["r", "S", "imp_vol", "strike_frac", "tau", "log_moneyness"]
CSV_PATH = "df_train.csv"

def report_percentiles(df: pd.DataFrame, regime: str):
    regime_data = df[df["market_regime"] == regime][COLUMNS_TO_DESCRIBE]
    percentiles = regime_data.quantile([0.05, 0.95])
    print(f"\n5th and 95th percentiles ({regime}):")
    for col in COLUMNS_TO_DESCRIBE:
        q5 = percentiles.loc[0.05, col]
        q95 = percentiles.loc[0.95, col]
        print(f"{col}: [{q5:.5f}, {q95:.5f}]")

if __name__ == "__main__":
    calibration = pd.read_csv(CSV_PATH)
    for regime in ["normal", "crisis"]:
        report_percentiles(calibration, regime)