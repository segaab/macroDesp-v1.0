# complacency_index_full_twochunk_part1.py
"""
Complacency Index — Diagnostics & Testing (Part 1)

This file is chunk 1/2 of the full Streamlit script. Put both chunks
in the same directory and run the combined content as a single script
(or paste chunk 2 right after chunk 1 to create one file).

Requirements:
  pip install streamlit pandas numpy yahooquery requests fredapi
Set FRED_API_KEY in your environment if you want FRED HTTP/fredapi calls to work.
"""
import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings

# yahooquery for price fetch
from yahooquery import Ticker

# optional fredapi
try:
    from fredapi import Fred
    HAS_FREDAPI = True
except Exception:
    Fred = None
    HAS_FREDAPI = False

# ---------------------------
# Use the user-provided fetch_price_data (kept verbatim / as requested)
# ---------------------------
def fetch_price_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data using yahooquery.
    
    Parameters
    ----------
    symbol : str
        The ticker symbol (e.g., "EURUSD=X", "^GSPC", "GC=F").
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    interval : str
        Data interval. Options: "1h", "1d", "1wk", etc. Default: "1d".
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with datetime index and columns:
        ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    try:
        ticker = Ticker(symbol)
        hist = ticker.history(start=start, end=end, interval=interval)

        if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
            # consistent with previous behavior: return empty df
            st.warning(f"No data returned for {symbol}.")
            return pd.DataFrame()

        # Reset index if multi-index
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.reset_index(level=0, drop=True)

        hist = hist.reset_index()

        # Normalize column names (case-insensitive handling)
        # keep mapping to the user's requested output names
        rename_map = {}
        for c in hist.columns:
            lc = c.lower()
            if lc == "open":
                rename_map[c] = "Open"
            elif lc == "high":
                rename_map[c] = "High"
            elif lc == "low":
                rename_map[c] = "Low"
            elif lc in ("close", "adjclose", "adjusted_close", "closeadj"):
                rename_map[c] = "Close"
            elif "volume" in lc:
                rename_map[c] = "Volume"
            elif lc == "date":
                rename_map[c] = "date"
        hist = hist.rename(columns=rename_map)

        # Ensure proper datetime column exists, else try index
        if "date" not in hist.columns:
            # try to detect datetime-like column
            for c in hist.columns:
                if np.issubdtype(hist[c].dtype, np.datetime64):
                    hist = hist.rename(columns={c: "date"})
                    break
        if "date" not in hist.columns:
            # fallback to index (after reset_index above, index likely contains positional integers)
            hist["date"] = pd.to_datetime(hist.index)

        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist.set_index("date").sort_index()

        # Ensure required cols exist; if something odd appears (DataFrame in col), coerce to 1D
        for req in ["Open", "High", "Low", "Close", "Volume"]:
            if req not in hist.columns:
                hist[req] = np.nan
            else:
                col = hist[req]
                if isinstance(col, pd.DataFrame):
                    # choose first numeric column
                    non_nan_counts = col.notna().sum()
                    if not non_nan_counts.empty:
                        best_col = non_nan_counts.idxmax()
                        warnings.warn(f"fetch_price_data: column '{req}' returned DataFrame; selecting '{best_col}'")
                        hist[req] = col[best_col]
                    else:
                        hist[req] = col.iloc[:, 0]

        # Coerce numeric types
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")

        # If no usable OHLCV, return empty frame
        if hist[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all").empty:
            st.warning(f"No usable OHLCV for {symbol}")
            return pd.DataFrame()

        return hist[["Open", "High", "Low", "Close", "Volume"]].copy()

    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------
# FRED helpers (fredapi preferred; HTTP fallback)
# ---------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
fred_client = None
if FRED_API_KEY and HAS_FREDAPI:
    try:
        fred_client = Fred(api_key=FRED_API_KEY)
    except Exception:
        fred_client = None

def fred_series_http(series_id: str, start: str, end: str, api_key: str) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "observation_start": start, "observation_end": end,
              "api_key": api_key, "file_type": "json"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        obs = j.get("observations", [])
        rows = []
        for o in obs:
            date = o.get("date")
            val = o.get("value")
            if val is None or val == ".":
                v = np.nan
            else:
                try:
                    v = float(val)
                except Exception:
                    v = np.nan
            rows.append((pd.to_datetime(date), v))
        if not rows:
            return pd.Series(dtype=float)
        s = pd.Series({d: v for d, v in rows})
        s.index = pd.to_datetime(s.index)
        s.name = series_id
        return s.sort_index()
    except Exception as e:
        st.warning(f"FRED HTTP fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float)

def get_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    if fred_client is not None:
        try:
            s = fred_client.get_series(series_id, observation_start=start, observation_end=end)
            s.index = pd.to_datetime(s.index)
            s.name = series_id
            return s.sort_index()
        except Exception:
            pass
    if FRED_API_KEY:
        return fred_series_http(series_id, start, end, FRED_API_KEY)
    st.warning("No FRED API key provided; returning empty series.")
    return pd.Series(dtype=float)

# ---------------------------
# Utility: safe 1D extraction to avoid pandas shape issues
# ---------------------------
def to_1d_array(obj, target_index):
    """
    Convert Series/DataFrame/ndarray to 1D numpy array aligned to target_index length.
    If DataFrame with multiple columns is provided, select the column with fewest NaNs.
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            ser = obj.iloc[:, 0]
        else:
            non_nan_counts = obj.notna().sum()
            best_col = non_nan_counts.idxmax()
            warnings.warn(f"to_1d_array: selecting column '{best_col}' from DataFrame")
            ser = obj[best_col]
    elif isinstance(obj, pd.Series):
        ser = obj
    else:
        arr = np.asarray(obj)
        if arr.ndim == 1:
            if arr.size == len(target_index):
                return arr
            if arr.size == 1:
                return np.repeat(arr.item(), len(target_index))
            return np.resize(arr, len(target_index))
        elif arr.ndim == 2:
            col = arr[:, 0]
            if col.size != len(target_index):
                return np.resize(col, len(target_index))
            return col
        else:
            return np.full(len(target_index), np.nan)

    ser = ser.reindex(target_index).copy()
    try:
        ser = pd.to_numeric(ser, errors="coerce")
    except Exception:
        pass
    arr = ser.to_numpy().ravel()
    if arr.size != len(target_index):
        arr = np.resize(arr, len(target_index))
    return arr

# ---------------------------
# Computation helpers (realized vol, rel-vol, zscore, mapping)
# ---------------------------
def compute_rel_volume(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["VolAvg50"] = df["Volume"].rolling(lookback, min_periods=1).mean().replace(0, np.nan).fillna(method="bfill").fillna(1.0)
    df["VolRel"] = df["Volume"] / df["VolAvg50"]
    return df

def realized_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    if "Close" not in df.columns:
        s_close = pd.Series(0.0, index=df.index)
    else:
        s_close = pd.to_numeric(df["Close"], errors="coerce").fillna(method="ffill").fillna(0.0)
    rets = s_close.pct_change().fillna(0.0)
    rv = rets.rolling(window, min_periods=1).std() * np.sqrt(252) * 100.0
    rv = rv.reindex(df.index)
    rv.name = "RealizedVol20"
    return rv.fillna(0.0)

def zscore(series: pd.Series, baseline_days: int = 252) -> pd.Series:
    s = pd.Series(series).astype(float)
    mean = s.shift(1).rolling(baseline_days, min_periods=30).mean()
    std = s.shift(1).rolling(baseline_days, min_periods=30).std().replace(0, np.nan)
    z = (s - mean) / std
    return z.fillna(0.0)

def map_to_0_10(x, cap: float = 6.0):
    return 10.0 * np.clip(x, 0.0, cap) / cap

def compute_NTS_from_headlines(neg_frac: float):
    if neg_frac < 0.2:
        return 10.0
    if neg_frac < 0.6:
        return 5.0
    return 0.0


# complacency_index_full_twochunk_part2.py
"""
Complacency Index — Diagnostics & Testing (Part 2)
Continuing from part 1: UI, main flow, charts, CSV download.
"""
# ---------------------------
# Streamlit UI (main)
# ---------------------------
st.set_page_config(layout="wide", page_title="Complacency Index Diagnostics")
st.title("Complacency Index — Diagnostics & Testing (Lehman / GFC focus)")

with st.sidebar:
    st.header("Controls")
    symbols_input = st.text_input("Symbols (comma-separated)", value="XLF,IYR,CL=F")
    start_date = st.date_input("Start date", value=datetime(2007, 1, 1))
    end_date = st.date_input("End date", value=datetime(2008, 12, 31))
    vol_lookback = st.number_input("Volume lookback (days)", value=50, min_value=5, step=5)
    rv_window = st.number_input("Realized vol window (days)", value=20, min_value=5, step=1)
    baseline_days = st.number_input("Z-score baseline days", value=252, min_value=60, step=1)
    complacency_threshold = st.slider("Complacency threshold (CI)", min_value=0.0, max_value=10.0, value=7.0, step=0.5)
    run_button = st.button("Fetch & compute Complacency Index")

symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

if run_button:
    if not symbols:
        st.error("Please specify at least one symbol.")
    else:
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")
        st.info(f"Fetching data for: {', '.join(symbols)} from {start} → {end}")

        # calendar
        cal = pd.date_range(start=start, end=end, freq="B")

        # Fetch price data
        price_data = {}
        for s in symbols:
            df = fetch_price_data(s, start, end, interval="1d")
            if df.empty:
                st.warning(f"No price data for {s}. Using synthetic fallback.")
                dates = cal
                n = len(dates)
                rng = np.random.default_rng(abs(hash(s)) % (2**32))
                price = 100.0 * np.exp(np.cumsum(rng.normal(loc=-0.0003, scale=0.03, size=n)))
                vol = rng.integers(500, 5000, size=n)
                df = pd.DataFrame({"Open": price, "High": price * 1.01, "Low": price * 0.99, "Close": price, "Volume": vol}, index=dates)
            # align and coerce
            df.index = pd.to_datetime(df.index)
            df = df.reindex(cal).ffill().bfill()
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).astype(float)
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce").fillna(method="ffill").fillna(0.0)
            price_data[s] = df

        st.success("Price data prepared.")

        # Per-symbol derived tables
        per_sym_tables = {}
        for sym, df in price_data.items():
            dfc = compute_rel_volume(df, lookback=vol_lookback)
            rv = realized_vol(dfc, window=rv_window)
            dfc["RealizedVol20"] = to_1d_array(rv, dfc.index)
            dfc["R5"] = dfc["Close"].pct_change(5).fillna(0.0)
            dfc["AbsR5"] = dfc["R5"].abs()
            per_sym_tables[sym] = dfc[["Close", "Volume", "VolAvg50", "VolRel", "RealizedVol20", "R5", "AbsR5"]].copy()

        # Fetch FRED series DGS10 and DRSFRMACBS
        st.info("Fetching FRED series (DGS10, DRSFRMACBS)...")
        dgs10 = get_fred_series("DGS10", start, end)
        drs = get_fred_series("DRSFRMACBS", start, end)

        # Align FRED series to business calendar and forward-fill where appropriate
        if not dgs10.empty:
            dgs10 = dgs10.reindex(cal).ffill().bfill()
        else:
            dgs10 = pd.Series(index=cal, data=np.nan, name="DGS10")
        if not drs.empty:
            drs = drs.reindex(cal).ffill().bfill()
        else:
            drs = pd.Series(index=cal, data=np.nan, name="DRSFRMACBS")

        # Build diagnostics DataFrame
        diag = pd.DataFrame(index=cal)
        diag["DGS10"] = to_1d_array(dgs10, diag.index)
        diag["DRSFRMACBS"] = to_1d_array(drs, diag.index)

        for sym, tab in per_sym_tables.items():
            diag[f"{sym}_Close"] = to_1d_array(tab["Close"], diag.index)
            diag[f"{sym}_VolRel"] = to_1d_array(tab["VolRel"], diag.index)
            diag[f"{sym}_RV20"] = to_1d_array(tab["RealizedVol20"], diag.index)
            diag[f"{sym}_R5"] = to_1d_array(tab["R5"], diag.index)
            diag[f"{sym}_AbsR5"] = to_1d_array(tab["AbsR5"], diag.index)

        # Compute z-scores for defaults & yields & price returns & vol
        diag["z_defaults"] = zscore(diag["DRSFRMACBS"], baseline_days=baseline_days)
        diag["z_yield"] = zscore(diag["DGS10"], baseline_days=baseline_days)

        # primary symbol price R5 z (negative so price declines increase FDS)
        primary_sym = symbols[0]
        diag["price_r5"] = diag.get(f"{primary_sym}_R5", pd.Series(index=diag.index, data=0.0))
        diag["z_price_r5"] = zscore(-diag["price_r5"], baseline_days=baseline_days)

        # stress z
        diag["stress_z"] = diag["z_defaults"].fillna(0.0) + diag["z_yield"].fillna(0.0)

        # realized vol z average (across symbols)
        rv_zs = []
        for s in symbols:
            col_rv = f"{s}_RV20"
            if col_rv in diag.columns:
                diag[f"{s}_z_rv"] = zscore(diag[col_rv], baseline_days=baseline_days)
                rv_zs.append(diag[f"{s}_z_rv"])
        if rv_zs:
            rv_z_df = pd.concat(rv_zs, axis=1).fillna(0.0)
            diag["avg_z_rv"] = rv_z_df.mean(axis=1)
        else:
            diag["avg_z_rv"] = 0.0

        # FDS
        diag["FDS_raw_z"] = (diag["z_defaults"].fillna(0.0) + diag["z_yield"].fillna(0.0) - diag["z_price_r5"].fillna(0.0)).clip(lower=0.0)
        diag["FDS"] = map_to_0_10(diag["FDS_raw_z"])

        # VFS (per-asset raw m, then aggregate)
        per_asset_m = []
        for s in symbols:
            volrel = pd.Series(diag.get(f"{s}_VolRel", pd.Series(index=diag.index, data=1.0))).fillna(1.0)
            absr5 = pd.Series(diag.get(f"{s}_AbsR5", pd.Series(index=diag.index, data=0.0))).fillna(0.0)
            m = (np.maximum(0.0, volrel - 1.0) / (1.0 + absr5)).fillna(0.0)
            diag[f"{s}_VFS_raw"] = to_1d_array(m, diag.index)
            per_asset_m.append(m)
        if per_asset_m:
            m_df = pd.concat(per_asset_m, axis=1)
            diag["VFS_raw"] = m_df.mean(axis=1)
        else:
            diag["VFS_raw"] = 0.0
        diag["VFS"] = map_to_0_10(diag["VFS_raw"], cap=4.0)

        # VSS
        diag["VSS_raw"] = (diag["stress_z"].fillna(0.0) - diag["avg_z_rv"].fillna(0.0)).clip(lower=0.0)
        diag["VSS"] = map_to_0_10(diag["VSS_raw"])

        # NTS via curated headlines (preserved)
        curated = {
            "2008-09-12": ["Markets jitter ahead of key bank meetings"],
            "2008-09-13": ["Banks seek capital as tensions rise"],
            "2008-09-14": ["Lehman Brothers files for bankruptcy protection"],
            "2008-09-15": ["Investors fear domino effect after Lehman folds"]
        }
        def headline_neg_frac_for_date(dt):
            dstr = dt.strftime("%Y-%m-%d")
            hs = curated.get(dstr, [])
            if not hs:
                return 0.0
            neg_count = 0
            for h in hs:
                low = h.lower()
                if any(k in low for k in ["bankrupt", "bankruptcy", "collapse", "panic", "default", "contagion", "fear", "fail", "fold"]):
                    neg_count += 1
            return neg_count / max(1, len(hs))
        diag["neg_frac"] = diag.index.to_series().apply(headline_neg_frac_for_date).values
        diag["NTS"] = diag["neg_frac"].apply(lambda nf: compute_NTS_from_headlines(nf))

        # Composite Complacency Index
        diag["ComplacencyIndex"] = (0.4 * diag["FDS"] + 0.2 * diag["VFS"] + 0.2 * diag["VSS"] + 0.2 * diag["NTS"])
        diag["ComplacencyFlag"] = (diag["ComplacencyIndex"] >= complacency_threshold)

        # Present results
        st.subheader("Diagnostics (tail)")
        st.dataframe(diag.tail(120))

        st.subheader("Complacency Index components (time series)")
        st.line_chart(diag[["ComplacencyIndex", "FDS", "VFS", "VSS", "NTS"]].fillna(0.0).tail(180))

        st.subheader("Macro series (10y yield & mortgage delinquency)")
        st.line_chart(diag[["DGS10", "DRSFRMACBS"]].fillna(method="ffill").tail(180))

        st.subheader("Per-symbol metrics (Close, VolRel, RealizedVol)")
        per_cols = []
        for s in symbols:
            per_cols += [f"{s}_Close", f"{s}_VolRel", f"{s}_RV20"]
        st.dataframe(diag[per_cols].tail(120))

        flagged = diag[diag["ComplacencyFlag"]]
        if not flagged.empty:
            st.warning(f"Complacency flagged on {len(flagged)} business days. First flagged date: {flagged.index[0].strftime('%Y-%m-%d')}")
            st.dataframe(flagged[["ComplacencyIndex", "FDS", "VFS", "VSS", "NTS"]].head(20))
        else:
            st.info("No Complacency flag hits in the selected range with current parameters.")

        csv = diag.reset_index().to_csv(index=False)
        st.download_button("Download diagnostics CSV", csv, file_name="complacency_diagnostics.csv", mime="text/csv")