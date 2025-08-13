"""
Processes the synthetic data sets so that they are ready to feed into the LSTM
"""
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

bs_n = pd.read_csv('bs_paths_normal.csv')
rv_n = pd.read_csv('rough_paths_normal.csv')
bs_c = pd.read_csv('bs_paths_crisis.csv')
rv_c = pd.read_csv('rough_paths_crisis.csv')

rv_n['inst_vol'] = np.sqrt(rv_n['V'])
rv_c['inst_vol'] = np.sqrt(rv_c['V'])

rv_n.drop('V', axis=1, inplace=True)
rv_c.drop('V', axis=1, inplace=True)

bs_n.rename(columns={'imp_vol': 'inst_vol'}, inplace=True)
bs_c.rename(columns={'imp_vol': 'inst_vol'}, inplace=True)

for df in [rv_n, rv_c, bs_n, bs_c]:
    df['tau_norm'] = df['tau_remain'] / df['tenor']

print("rv Crisis columns:", rv_c.columns.tolist())
print("rv Normal columns:", rv_n.columns.tolist())
print("BS Crisis columns:", bs_c.columns.tolist())
print("BS Normal columns:", bs_n.columns.tolist())

bs = pd.concat([bs_n, bs_c], ignore_index=True)
rv = pd.concat([rv_n, rv_c], ignore_index=True)

bs.to_csv("bs_paths_combined.csv", index=False)
rv.to_csv("rough_paths_combined.csv", index=False)
bs_n.to_csv("bs_paths_normal.csv", index =False)
bs_c.to_csv("bs_paths_crisis.csv",index=False)
rv_n.to_csv("rough_paths_normal.csv",index = False)
rv_c.to_csv("rough_paths_crisis.csv",index = False)

print("Combined BS shape:", bs.shape)
print("Combined rv shape:", rv.shape)
