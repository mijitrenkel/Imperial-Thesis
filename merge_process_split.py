"""
merge_process_split.py
This code imports and processes multiple datasets of the S&P 500 call options:
- Option metrics data
- S&P 500 daily closing prices
- Risk-free rate data
Processing steps include:
1. Dropping rows with missing or invalid values in key columns 
   (implied volatility, delta, volume)
2. Converting 'date' and 'exdate' columns to datetime format
3. Renaming columns across datasets for consistency and dropping irrelevant columns
4. Merging the options, underlying index prices, and risk-free rate data 
   into a single DataFrame based on date
5. Calculating additional features:
   - 'tau': time to expiry in business days
   - 'strike_frac':strike / underlying spot
   - 'moneyness: the options moneyness 
6. Assigning market regime labels to each row based on date ranges.
7. Splitting the processed dataset into train, validation, and test sets.
"""

import pandas as pd
import numpy as np

opx = pd.read_csv('data.csv') #Option metrics data
closing = pd.read_csv('sp500_close.csv') #SP500 daily closing prices
r = pd.read_csv('r.csv') #Risk-free rate data 
r['rate'] = r['rate'] / 100.0

opx.dropna(subset=['impl_volatility', 'delta', 'volume'], inplace=True)
opx = opx[opx["volume"]>0].copy()
opx['date'] = pd.to_datetime(opx['date'], dayfirst=True, format='%d/%m/%Y')
opx['exdate'] = pd.to_datetime(opx['exdate'], dayfirst=True, format='%d/%m/%Y')
closing['Date'] = pd.to_datetime(closing['Date'], dayfirst=True, format='%Y-%m-%d')
r['date'] = pd.to_datetime(r['date'], dayfirst=True, format='%Y-%m-%d')
r = r.rename(columns={'days': 'tau', 'rate': 'r'})

#Drop irrelevant columns
to_drop = ['secid', 'cp_flag', 'exercise_style', 'index_flag', 'issuer', 'optionid','expiry_indicator',
           'contract_size', 'forward_price', 'gamma', 'vega', 'theta']
opx = opx.drop(columns=[c for c in to_drop if c in opx.columns])

#Merge option data with S&P500 closing prices
merged = opx.merge(closing.rename(columns={'Date': 'date', 'Close': 'spx_close'}),
on='date',how='left')

#Calculate 'tau' and 'strike_frac'
merged["tau"] = (merged["exdate"] - merged["date"]).dt.days.astype(float).round(2)
merged["strike_price"] = pd.to_numeric(merged["strike_price"], errors='coerce').round(2)
merged["spx_close"] = pd.to_numeric(merged["spx_close"], errors='coerce').round(2)
merged["strike_frac"] = ((merged["strike_price"] / 1000) / merged["spx_close"]).round(2)
merged["log_moneyness"] = np.log(merged["spx_close"]/(merged["strike_price"]/1000))

#Merge with risk-free rate
merged = merged.merge(r[['date', 'tau', 'r']],on=['date', 'tau'],how='left')

#Function to fill missing values in risk-free rate
curve_dict = {d: grp.reset_index(drop=True) for d, grp in r.groupby('date')}
def fill_nearest(row):
    if not np.isnan(row['r']):
        return row['r']
    curve = curve_dict.get(row['date'])
    if curve is None or curve.empty:
        return np.nan
    idx = (curve['tau'] - row['tau']).abs().idxmin()
    return curve.at[idx, 'r']
mask = merged['r'].isna()
merged.loc[mask, 'r'] = merged[mask].apply(fill_nearest, axis=1)
merged['r'] = merged['r'].fillna(method='bfill')

merged.rename(columns={'spx_close': 'S','impl_volatility': 'imp_vol'}, inplace=True)

#Function to assign market regimes
def assign_market_regime(date):
    if pd.Timestamp('2007-10-01') <= date <= pd.Timestamp('2009-03-31'):
        return 'crisis'
    elif pd.Timestamp('2020-02-15') <= date <= pd.Timestamp('2020-04-30'):
        return 'covid'
    elif pd.Timestamp('2022-01-01') <= date <= pd.Timestamp('2022-12-31'):
        return 'hike2022'
    else:
        return 'normal'

merged['market_regime'] = merged['date'].apply(assign_market_regime)
#Add moneyness classification
def moneyness_bucket(x):
    if pd.isna(x):
        return np.nan
    if   x >  +0.20: return "Deep ITM"
    elif x > +0.05:  return "ITM"
    elif x >= -0.05: return "ATM"
    elif x >= -0.20: return "OTM"
    else:            return "Deep OTM"

merged["moneyness"] = merged["log_moneyness"].apply(moneyness_bucket)

#Splitting data into train,val,test
data_train = merged[(merged['date'] <= '2015-12-31')]
data_train.to_csv('data_train.csv', index=False)

data_val = merged[(merged['date'] >= '2016-01-01') & (merged['date'] <= '2018-12-31')]
data_val.to_csv('data_val.csv', index=False)

data_test = merged[merged['date'] >= '2019-01-01']
data_test.to_csv('data_test.csv', index=False)

