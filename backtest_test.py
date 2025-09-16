# ---------------------------
# GFC Short-only Backtest Dashboard
# ---------------------------

# ---------------------------
# Standard libraries
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ---------------------------
# @dataclass placeholders (assume already present)
# ---------------------------
# @dataclass
# class MarketSnapshot:
#     date: pd.Timestamp
#     asset_prices: dict
#     asset_volumes: dict
#     headline_scores: dict

# ---------------------------
# Config / Scenario (GFC)
# ---------------------------
SCENARIO = {
    "preset": "GFC",
    "label": "GFC (Lehman bankruptcy - Sep 15, 2008 → Mar 31, 2009)",
    "start": "2008-09-14",  # day before Lehman collapse
    "end": "2009-03-31",
    "tag": "gfc2008"
}

# Sector -> asset proxies
ASSET_MAP = {
    "Energy": ["CL"],
    "Financials": ["XLF"],
    "Industrials": ["XLI"],
    "Consumer Discretionary": ["XLY"],
    "Materials": ["XLB"],
    "Real Estate": ["XLRE"],
    "Technology": ["NQ"]
}

ROOT = Path(".")
DATA_ROOT = ROOT / "data_placeholder"
DOCS = ROOT / "docs"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Helpers: synthetic data
# ---------------------------
def generate_synthetic(asset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = DATA_ROOT / f"{asset}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df[(df["date"] >= start) & (df["date"] <= end)].copy()

    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    rng = np.random.RandomState(sum(bytearray(asset.encode())) % 2**32)
    mu = -0.0008 if asset in ("XLF","XLRE") else -0.0003
    sigma = 0.03
    returns = rng.normal(loc=mu, scale=sigma, size=n)
    price = 100.0 * np.exp(np.cumsum(returns))
    vol = rng.randint(500, 5000, size=n)
    high = price * (1 + rng.rand(n) * 0.02)
    low = price * (1 - rng.rand(n) * 0.02)
    open_ = price * (1 + rng.normal(0, 0.005, size=n))
    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": price,
        "volume": vol
    })
    df.to_parquet(path)
    return df

def load_price_matrix(selected_assets, start, end):
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    calendar = pd.date_range(start=start_ts, end=end_ts, freq="B")
    price_data = {}
    for asset in selected_assets:
        df = generate_synthetic(asset, start_ts, end_ts)
        df = df.set_index("date").sort_index()
        df = df.reindex(calendar).ffill().bfill().reset_index().rename(columns={"index": "date"})
        price_data[asset] = df
    return price_data, calendar

# ---------------------------
# Momentum breakdown signal
# ---------------------------
def momentum_breakdown_signal(df: pd.DataFrame, idx: int) -> int:
    if idx < 50:
        return 0
    window = df.iloc[max(0, idx-100):idx+1]
    today = df.iloc[idx]
    ma50 = window["close"].rolling(50, min_periods=1).mean().iloc[-1]
    vol50 = window["volume"].rolling(50, min_periods=1).mean().iloc[-1]
    vol_spike = today["volume"] > vol50 * 1.5
    falling = today["close"] < df.iloc[idx-3]["close"] if idx >= 3 else True
    return 1 if (today["close"] < ma50 and vol_spike and falling) else 0

# ---------------------------
# Backtest engine
# ---------------------------
class SimpleShortBacktest:
    def __init__(self, initial_capital=1_000_000, slippage_bps=30, margin_rate=0.1, allocation_pct=0.50):
        self.initial_capital = initial_capital
        self.slippage = slippage_bps / 10000.0
        self.margin_rate = margin_rate
        self.allocation_pct = allocation_pct

    def run(self, price_data: dict, calendar: pd.DatetimeIndex):
        assets = list(price_data.keys())
        positions = {a: 0 for a in assets}
        trades = []
        cash = float(self.initial_capital)
        equity_series = []

        closes = {a: price_data[a]["close"].values for a in assets}
        df_by_asset = {a: price_data[a] for a in assets}

        for i, date in enumerate(calendar):
            signals = {a: momentum_breakdown_signal(df_by_asset[a], i) for a in assets}
            signaled = [a for a, s in signals.items() if s == 1]
            target_positions = {a: 0 for a in assets}
            if signaled:
                total_short_notional = self.allocation_pct * self.initial_capital
                per_asset_notional = total_short_notional / len(signaled)
                for a in signaled:
                    price = closes[a][i]
                    units = int(per_asset_notional / price) if price > 0 else 0
                    target_positions[a] = -units

            for a in assets:
                cur, tgt = positions[a], target_positions[a]
                if cur != tgt:
                    price = closes[a][i]
                    exec_price = price * (1 - self.slippage) if tgt < cur else price * (1 + self.slippage)
                    cash += (cur - tgt) * exec_price
                    trades.append({
                        "date": pd.to_datetime(date),
                        "asset": a,
                        "from_pos": int(cur),
                        "to_pos": int(tgt),
                        "exec_price": float(exec_price),
                        "units": int(tgt - cur)
                    })
                    positions[a] = int(tgt)

            mtm = cash + sum([positions[a] * float(closes[a][i]) for a in assets])
            required_margin = sum([abs(positions[a]) * float(closes[a][i]) * self.margin_rate for a in assets])
            if mtm < required_margin:
                shorts = [(a, abs(positions[a]) * float(closes[a][i])) for a in assets if positions[a] < 0]
                if shorts:
                    shorts.sort(key=lambda x: x[1], reverse=True)
                    a_to_cover, _ = shorts[0]
                    cur = positions[a_to_cover]
                    cover_units = int(abs(cur) * 0.5)
                    if cover_units > 0:
                        exec_price = float(closes[a_to_cover][i]) * (1 + self.slippage)
                        new_pos = cur + cover_units
                        cash += (cur - new_pos) * exec_price
                        trades.append({
                            "date": pd.to_datetime(date),
                            "asset": a_to_cover,
                            "from_pos": int(cur),
                            "to_pos": int(new_pos),
                            "exec_price": float(exec_price),
                            "units": int(new_pos - cur)
                        })
                        positions[a_to_cover] = int(new_pos)
                        mtm = cash + sum([positions[a]*float(closes[a][i]) for a in assets])

            equity_series.append({"date": pd.to_datetime(date), "equity": mtm, "required_margin": required_margin})

        equity_df = pd.DataFrame(equity_series).set_index("date")
        metrics = self._compute_metrics(equity_df["equity"])
        trades_df = pd.DataFrame(trades)
        return {"equity": equity_df, "metrics": metrics, "trades": trades_df}

    def _compute_metrics(self, equity_series: pd.Series):
        if equity_series.empty:
            return {}
        net_pnl = float(equity_series.iloc[-1] - equity_series.iloc[0])
        peak = equity_series.cummax()
        dd = (equity_series - peak) / peak
        max_dd = float(dd.min())
        days = (equity_series.index[-1] - equity_series.index[0]).days or 1
        annualized = ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (365.0 / days) - 1) if equity_series.iloc[0] > 0 else 0.0
        return {"start_equity": float(equity_series.iloc[0]), "end_equity": float(equity_series.iloc[-1]),
                "net_pnl": net_pnl, "max_drawdown": max_dd, "annualized_return": annualized}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide", page_title="GFC Short-only Backtest")
st.title("Lehman / GFC Short-only Backtest — Morning Model Prereq")

with st.sidebar:
    st.header("Lehman (GFC) controls")
    st.markdown(f"**Scenario:** {SCENARIO['label']}")
    st.write(f"Date range: {SCENARIO['start']} → {SCENARIO['end']}")
    sectors = st.multiselect("Sectors (short-only proxies)", options=list(ASSET_MAP.keys()),
                             default=["Financials", "Real Estate", "Energy"])
    initial_cap = st.number_input("Initial capital (USD)", value=1_000_000, step=100_000)
    slippage_bps = st.number_input("Slippage (bps)", value=50)
    margin_rate = st.number_input("Margin rate (fraction)", value=0.12)
    run_bt = st.button("Run Lehman / GFC backtest")