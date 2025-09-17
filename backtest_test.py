# complacency_index_diagnostics_fixed.py
"""
Diagnostics + Complacency Index test app (fixed).
- Uses yahooquery fetch wrapper (user-provided style)
- Uses FRED (fredapi if available, else HTTP fallback)
- Computes rel-vol, realized vol (fixed to return Series), z-scores, and Complacency Index
"""
import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import requests

# yahooquery is required for price fetch
from yahooquery import Ticker

# fredapi optional
try:
    from fredapi import Fred
    HAS_FREDAPI = True
except Exception:
    Fred = None
    HAS_FREDAPI = False

# ---------------------------
# User-provided yahoo fetch function (kept as requested)
# ---------------------------
def fetch_price_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data using yahooquery. Returns DataFrame indexed by date with columns:
    ['Open','High','Low','Close','Volume'].
    """
    try:
        ticker = Ticker(symbol)
        hist = ticker.history(start=start, end=end, interval=interval)

        if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
            st.warning(f"No data returned for {symbol}.")
            return pd.DataFrame()

        # Reset index if multi-index (drop symbol level)
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.reset_index(level=0, drop=True)

        hist = hist.reset_index()

        # unify column names (handle adjclose vs close differences)
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

        hist = hist.rename(columns=rename_map)

        # ensure date column
        if "date" not in hist.columns:
            # try to detect a datetime-like column
            for c in hist.columns:
                if np.issubdtype(hist[c].dtype, np.datetime64):
                    hist = hist.rename(columns={c: "date"})
                    break
        if "date" not in hist.columns:
            hist["date"] = pd.to_datetime(hist.index)

        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.set_index("date").sort_index()

        # ensure required cols exist
        for req in ["Open", "High", "Low", "Close", "Volume"]:
            if req not in hist.columns:
                hist[req] = np.nan

        return hist[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------
# FRED helpers
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
    params = {"series_id": series_id, "observation_start": start, "observation_end": end, "api_key": api_key, "file_type": "json"}
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
# Computation helpers (fixed realized_vol returns Series)
# ---------------------------
def compute_rel_volume(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["VolAvg50"] = df["Volume"].rolling(lookback, min_periods=1).mean().replace(0, np.nan).fillna(method="bfill").fillna(1.0)
    df["VolRel"] = df["Volume"] / df["VolAvg50"]
    return df

def realized_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Return a Series (indexed like df.index) with annualized realized volatility (%).
    Always returns a Series to avoid pandas multi-column assignment errors.
    """
    # guard: if Close missing, create a flat series
    if "Close" not in df.columns:
        s_close = pd.Series(0.0, index=df.index)
    else:
        s_close = df["Close"].astype(float).copy()
    rets = s_close.pct_change().fillna(0.0)
    rv = rets.rolling(window, min_periods=1).std() * np.sqrt(252) * 100.0
    rv = rv.reindex(df.index)  # ensure index alignment
    rv.name = "RealizedVol20"
    return rv.fillna(0.0)

def zscore(series: pd.Series, baseline_days: int = 252) -> pd.Series:
    s = series.copy().astype(float)
    mean = s.shift(1).rolling(baseline_days, min_periods=30).mean()
    std = s.shift(1).rolling(baseline_days, min_periods=30).std().replace(0, np.nan)
    z = (s - mean) / std
    return z.fillna(0.0)

def map_to_0_10(x, cap: float = 6.0):
    return 10.0 * np.clip(x, 0.0, cap) / cap

# Sub-score helpers
def compute_FDS(defaults_z: pd.Series, yield_z: pd.Series, price_r5_z: pd.Series):
    raw = (defaults_z.fillna(0.0) + yield_z.fillna(0.0) - price_r5_z.fillna(0.0)).clip(lower=0.0)
    return map_to_0_10(raw)

def compute_VFS(vol_rel: pd.Series, r5_abs: pd.Series):
    m = (np.maximum(0.0, vol_rel - 1.0) / (1.0 + r5_abs)).fillna(0.0)
    return map_to_0_10(m, cap=4.0)

def compute_VSS(stress_z: pd.Series, vol_z: pd.Series):
    raw = (stress_z.fillna(0.0) - vol_z.fillna(0.0)).clip(lower=0.0)
    return map_to_0_10(raw)

def compute_NTS_from_finbert(neg_frac: float):
    if neg_frac < 0.2:
        return 10.0
    if neg_frac < 0.6:
        return 5.0
    return 0.0

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide", page_title="Complacency Index Diagnostics (fixed)")
st.title("Complacency Index — Diagnostics & Testing (fixed)")

with st.sidebar:
    st.header("Controls")
    symbols_input = st.text_input("Symbols (comma-separated)", value="XLF,IYR,CL=F")
    start_date = st.date_input("Start date", value=datetime(2007,1,1))
    end_date = st.date_input("End date", value=datetime(2008,12,31))
    vol_lookback = st.number_input("Volume lookback (days)", value=50, min_value=5, step=5)
    rv_window = st.number_input("Realized vol window (days)", value=20, min_value=5, step=1)
    baseline_days = st.number_input("Z-score baseline days", value=252, min_value=60, step=1)
    run_button = st.button("Fetch data & compute index")

symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

if run_button:
    if not symbols:
        st.error("Please specify at least one symbol.")
    else:
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")

        st.info(f"Fetching price data for: {', '.join(symbols)} from {start} → {end}")
        price_data = {}
        for sym in symbols:
            df = fetch_price_data(sym, start, end, interval="1d")
            if df.empty:
                st.warning(f"No price data for {sym} (empty). Using synthetic fallback.")
                dates = pd.date_range(start=start, end=end, freq="B")
                n = len(dates)
                rng = np.random.default_rng(abs(hash(sym)) % (2**32))
                price = 100.0 * np.exp(np.cumsum(rng.normal(loc=-0.0003, scale=0.03, size=n)))
                vol = rng.integers(500, 5000, n)
                df = pd.DataFrame({"Open": price, "High": price*1.01, "Low": price*0.99, "Close": price, "Volume": vol}, index=dates)
            df.index = pd.to_datetime(df.index)
            if "Volume" not in df.columns:
                df["Volume"] = 0
            if "Close" not in df.columns:
                df["Close"] = df["Open"].fillna(method="ffill")
            cal = pd.date_range(start=start, end=end, freq="B")
            df = df.reindex(cal).ffill().bfill()
            price_data[sym] = df

        st.success("Price data fetched / prepared.")

        # per-symbol derived tables
        per_sym_tables = {}
        for sym, df in price_data.items():
            dfc = compute_rel_volume(df, lookback=vol_lookback)
            # realized_vol returns a Series aligned to df index
            rv_series = realized_vol(dfc, window=rv_window)
            # assign safely using .values (index-aligned)
            dfc["RealizedVol20"] = rv_series.reindex(dfc.index).values
            dfc["R5"] = dfc["Close"].pct_change(5).fillna(0.0)
            dfc["AbsR5"] = dfc["R5"].abs()
            per_sym_tables[sym] = dfc[["Close","Volume","VolAvg50","VolRel","RealizedVol20","R5","AbsR5"]].copy()

        # fetch FRED series
        st.info("Fetching FRED series (DGS10, DRSFRMACBS)...")
        dgs10 = get_fred_series("DGS10", start, end)
        drs = get_fred_series("DRSFRMACBS", start, end)

        cal = pd.date_range(start=start, end=end, freq="B")
        if not dgs10.empty:
            dgs10 = dgs10.reindex(cal).ffill().bfill()
        else:
            dgs10 = pd.Series(index=cal, data=np.nan, name="DGS10")
        if not drs.empty:
            drs = drs.reindex(cal).ffill().bfill()
        else:
            drs = pd.Series(index=cal, data=np.nan, name="DRSFRMACBS")

        # diagnostics DataFrame
        diag = pd.DataFrame(index=cal)
        diag["DGS10"] = dgs10.values
        diag["DRSFRMACBS"] = drs.values

        for sym, tab in per_sym_tables.items():
            diag[f"{sym}_Close"] = tab["Close"].values
            diag[f"{sym}_VolRel"] = tab["VolRel"].values
            diag[f"{sym}_RV20"] = tab["RealizedVol20"].values
            diag[f"{sym}_R5"] = tab["R5"].values
            diag[f"{sym}_AbsR5"] = tab["AbsR5"].values

        # compute z-scores
        diag["z_defaults"] = zscore(diag["DRSFRMACBS"], baseline_days=baseline_days)
        diag["z_yield"] = zscore(diag["DGS10"], baseline_days=baseline_days)

        primary_sym = symbols[0]
        diag["price_r5"] = diag.get(f"{primary_sym}_R5", pd.Series(index=diag.index, data=0.0))
        diag["z_price_r5"] = zscore(-diag["price_r5"], baseline_days=baseline_days)

        diag["stress_z"] = diag["z_defaults"].fillna(0.0) + diag["z_yield"].fillna(0.0)

        # realized vol z (avg across symbols)
        rv_zs = []
        for sym in symbols:
            col_rv = f"{sym}_RV20"
            if col_rv in diag.columns:
                diag[f"{sym}_z_rv"] = zscore(diag[col_rv], baseline_days=baseline_days)
                rv_zs.append(diag[f"{sym}_z_rv"])
        if rv_zs:
            rv_z_df = pd.concat(rv_zs, axis=1).fillna(0.0)
            diag["avg_z_rv"] = rv_z_df.mean(axis=1)
        else:
            diag["avg_z_rv"] = 0.0

        # FDS
        diag["FDS_raw_z"] = (diag["z_defaults"].fillna(0.0) + diag["z_yield"].fillna(0.0) - diag["z_price_r5"].fillna(0.0)).clip(lower=0.0)
        diag["FDS"] = map_to_0_10(diag["FDS_raw_z"])

        # VFS raw per asset and aggregated
        per_asset_m = []
        for sym in symbols:
            volrel = diag.get(f"{sym}_VolRel", pd.Series(index=diag.index, data=1.0)).fillna(1.0)
            absr5 = diag.get(f"{sym}_AbsR5", pd.Series(index=diag.index, data=0.0)).fillna(0.0)
            m = (np.maximum(0.0, volrel - 1.0) / (1.0 + absr5)).fillna(0.0)
            diag[f"{sym}_VFS_raw"] = m
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

        # NTS using hand-curated headlines mapping (simple negative keyword fraction)
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
        diag["NTS"] = diag["neg_frac"].apply(lambda nf: compute_NTS_from_finbert(nf))

        diag["ComplacencyIndex"] = (0.4 * diag["FDS"] + 0.2 * diag["VFS"] + 0.2 * diag["VSS"] + 0.2 * diag["NTS"])
        diag["ComplacencyFlag"] = (diag["ComplacencyIndex"] >= 7.0)

        # Display outputs
        st.subheader("Diagnostics (last 120 rows)")
        st.dataframe(diag.tail(120))

        st.subheader("Complacency Index components (time series)")
        st.line_chart(diag[["ComplacencyIndex","FDS","VFS","VSS","NTS"]].fillna(0.0).tail(180))

        st.subheader("Macro series")
        st.line_chart(diag[["DGS10","DRSFRMACBS"]].fillna(method="ffill").tail(180))

        st.subheader("Per-symbol metrics (Close, VolRel, RV20)")
        cols_to_show = []
        for sym in symbols:
            cols_to_show += [f"{sym}_Close", f"{sym}_VolRel", f"{sym}_RV20"]
        st.dataframe(diag[cols_to_show].tail(120))

        flagged = diag[diag["ComplacencyFlag"]]
        if not flagged.empty:
            st.warning(f"Complacency flagged on {len(flagged)} business days. First flagged date: {flagged.index[0].strftime('%Y-%m-%d')}")
            st.dataframe(flagged[["ComplacencyIndex","FDS","VFS","VSS","NTS"]].head(20))
        else:
            st.info("No Complacency flag hits in the selected range with current parameters.")

        csv = diag.reset_index().to_csv(index=False)
        st.download_button("Download diagnostics CSV", csv, file_name="complacency_diagnostics.csv", mime="text/csv")