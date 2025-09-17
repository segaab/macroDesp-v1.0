import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
from fredapi import Fred
import os

# --- Load API keys ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# --- Functions ---
def fetch_yahoo_ohlcv(symbol, start, end):
    t = Ticker(symbol)
    df = t.history(start=start, end=end)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    df = df[['open','high','low','close','volume']].copy()
    df = df.rename(columns={'close':'adjclose'})  # unify naming
    return df

def compute_rel_volume(df, lookback=50):
    df['vol_avg'] = df['volume'].rolling(lookback, min_periods=1).mean()
    df['vol_rel'] = df['volume'] / df['vol_avg']
    return df

def realized_vol(df, window=20):
    rets = df['adjclose'].pct_change().fillna(0)
    rv = rets.rolling(window, min_periods=1).std() * np.sqrt(252) * 100
    df['realized_vol20'] = rv
    return df

def zscore(series, baseline_days=252):
    mean = series.rolling(baseline_days, min_periods=30).mean()
    std = series.rolling(baseline_days, min_periods=30).std().replace(0, np.nan)
    return (series - mean) / std

# --- Data Fetch ---
start = "2007-01-01"
end = "2008-12-31"

st.title("Complacency Index Variable Diagnostics")

# Price & Volume
xlf = fetch_yahoo_ohlcv("XLF", start, end)
xlf = compute_rel_volume(xlf)
xlf = realized_vol(xlf)

# Add z-scores
xlf['z_return5'] = zscore(xlf['adjclose'].pct_change(5))

# FRED: 10y yield
dgs10 = fred.get_series("DGS10", observation_start=start, observation_end=end)
dgs10 = dgs10.to_frame("DGS10")
dgs10.index = pd.to_datetime(dgs10.index)

# FRED: mortgage delinquency
drs = fred.get_series("DRSFRMACBS", observation_start=start, observation_end=end)
drs = drs.to_frame("DRSFRMACBS")
drs.index = pd.to_datetime(drs.index)

# Forward fill quarterly → daily
drs_daily = drs.reindex(pd.date_range(start, end, freq="B")).ffill()

# Merge all data
df = pd.concat([xlf, dgs10, drs_daily], axis=1)

# Add z-scores for macro
df['z_defaults'] = zscore(df['DRSFRMACBS'])
df['z_yield'] = zscore(df['DGS10'])

# Identical close detector
df['identical_close_flag'] = df['adjclose'].diff().eq(0).astype(int)

st.subheader("Diagnostics DataFrame")
st.dataframe(df.tail(50))

# Optionally add charts
st.line_chart(df[['adjclose','vol_rel','realized_vol20']])
st.line_chart(df[['DGS10','DRSFRMACBS']])