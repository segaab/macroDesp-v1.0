import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
from fredapi import Fred
import os

# --- Load API keys ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# --- Yahooquery fetch function ---
def fetch_price_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data using yahooquery.
    """
    try:
        ticker = Ticker(symbol)
        hist = ticker.history(start=start, end=end, interval=interval)

        if hist.empty:
            print(f"No data returned for {symbol}.")
            return pd.DataFrame()

        # Reset index if multi-index
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.reset_index(level=0, drop=True)

        hist = hist.reset_index()
        hist = hist.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })

        # Ensure proper datetime
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.set_index("date")

        return hist[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# --- Helpers ---
def compute_rel_volume(df, lookback=50):
    df['VolAvg'] = df['Volume'].rolling(lookback, min_periods=1).mean()
    df['VolRel'] = df['Volume'] / df['VolAvg']
    return df

def realized_vol(df, window=20):
    rets = df['Close'].pct_change().fillna(0)
    rv = rets.rolling(window, min_periods=1).std() * np.sqrt(252) * 100
    df['RealizedVol20'] = rv
    return df

def zscore(series, baseline_days=252):
    mean = series.rolling(baseline_days, min_periods=30).mean()
    std = series.rolling(baseline_days, min_periods=30).std().replace(0, np.nan)
    return (series - mean) / std

# --- Dates ---
start = "2007-01-01"
end = "2008-12-31"

st.title("Complacency Index Variable Diagnostics")

# --- Price & Volume (XLF example) ---
xlf = fetch_price_data("XLF", start, end, "1d")

if not xlf.empty:
    xlf = compute_rel_volume(xlf)
    xlf = realized_vol(xlf)
    xlf['ZReturn5'] = zscore(xlf['Close'].pct_change(5))
    xlf['IdenticalCloseFlag'] = xlf['Close'].diff().eq(0).astype(int)
else:
    st.error("Yahooquery returned no data for XLF")

# --- FRED: 10y yield ---
dgs10 = fred.get_series("DGS10", observation_start=start, observation_end=end)
dgs10 = dgs10.to_frame("DGS10")
dgs10.index = pd.to_datetime(dgs10.index)

# --- FRED: mortgage delinquency ---
drs = fred.get_series("DRSFRMACBS", observation_start=start, observation_end=end)
drs = drs.to_frame("DRSFRMACBS")
drs.index = pd.to_datetime(drs.index)

# Forward fill quarterly → daily
drs_daily = drs.reindex(pd.date_range(start, end, freq="B")).ffill()

# --- Merge all data ---
df = pd.concat([xlf, dgs10, drs_daily], axis=1)

# Add z-scores for macro
df['ZDefaults'] = zscore(df['DRSFRMACBS'])
df['ZYield'] = zscore(df['DGS10'])

# --- Display ---
st.subheader("Diagnostics DataFrame")
st.dataframe(df.tail(50))

# --- Charts ---
if not xlf.empty:
    st.line_chart(df[['Close','VolRel','RealizedVol20']])
st.line_chart(df[['DGS10','DRSFRMACBS']])