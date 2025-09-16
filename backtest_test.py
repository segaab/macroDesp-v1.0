# backtest_gfc_onefile.py
"""
All-in-one Streamlit backtest for the Lehman / GFC scenario (short-only).
Dependencies:
  pip install streamlit pandas numpy pyyaml pyarrow reportlab
Run:
  streamlit run backtest_gfc_onefile.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------------------
# Config / Scenario (GFC)
# ---------------------------
SCENARIO = {
    "preset": "GFC",
    "label": "GFC (Lehman bankruptcy - Sep 15, 2008 → Mar 31, 2009)",
    "start": "2008-09-15",
    "end": "2009-03-31",
    "tag": "gfc2008"
}

# Sector -> asset proxies
ASSET_MAP = {
    "Energy": ["CL"],                   # crude futures proxy
    "Financials": ["XLF"],              # financial ETF proxy
    "Industrials": ["XLI"],
    "Consumer Discretionary": ["XLY"],
    "Materials": ["XLB"],
    "Real Estate": ["XLRE"],
    "Technology": ["NQ"]                # Nasdaq futures proxy (tech)
}

# Output paths
ROOT = Path(".")
DATA_ROOT = ROOT / "data_placeholder"
DOCS = ROOT / "docs"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Utilities: data generation / loader
# ---------------------------
def generate_synthetic(asset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Generate synthetic daily business-day OHLCV for asset and persist to parquet for caching."""
    path = DATA_ROOT / f"{asset}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df[(df["date"] >= start) & (df["date"] <= end)].copy()

    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    # seed to make reproducible per asset
    rng = np.random.RandomState(sum(bytearray(asset.encode())) % 2**32)
    # drift negative to emulate GFC down pressure for risk assets (but keep variety)
    mu = -0.0008 if asset in ("XLF","XLRE") else -0.0003
    sigma = 0.03
    returns = rng.normal(loc=mu, scale=sigma, size=n)
    price = 100.0 * np.exp(np.cumsum(returns))  # geometric walk
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
    """Return dict of asset: DataFrame reindexed to common business-day index with ffill for missing days."""
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    calendar = pd.date_range(start=start_ts, end=end_ts, freq="B")
    price_data = {}
    for asset in selected_assets:
        df = generate_synthetic(asset, start_ts, end_ts)
        df = df.set_index("date").sort_index()
        # reindex to full calendar and forward-fill last known values (reasonable for synthetic)
        df = df.reindex(calendar).ffill().bfill().reset_index().rename(columns={"index": "date"})
        price_data[asset] = df
    return price_data, calendar

# ---------------------------
# Strategy: momentum_breakdown_signal
# ---------------------------
def momentum_breakdown_signal(df: pd.DataFrame, idx: int) -> int:
    """
    Simple short trigger:
    - close below 50-day MA
    - volume spike > 1.5 * 50-day avg volume
    - close falling vs 3 days ago
    Returns 1 for short signal, 0 otherwise.
    df assumed indexed by date ascending.
    idx is integer position into df (0..len-1).
    """
    if idx < 50:
        return 0
    window = df.iloc[max(0, idx-100):idx+1]  # use up to 100 lookback to compute averages robustly
    today = df.iloc[idx]
    ma50 = window["close"].rolling(50, min_periods=1).mean().iloc[-1]
    vol50 = window["volume"].rolling(50, min_periods=1).mean().iloc[-1]
    vol_spike = today["volume"] > vol50 * 1.5
    falling = True
    if idx >= 3:
        falling = today["close"] < df.iloc[idx-3]["close"]
    if (today["close"] < ma50) and vol_spike and falling:
        return 1
    return 0

# ---------------------------
# Backtest engine
# ---------------------------
class SimpleShortBacktest:
    def __init__(self, initial_capital=1_000_000, slippage_bps=30, margin_rate=0.1):
        """
        slippage_bps: basis points (e.g., 30 -> 0.003)
        margin_rate: maintenance margin fraction applied to notional of open shorts
        """
        self.initial_capital = initial_capital
        self.slippage = slippage_bps / 10000.0
        self.margin_rate = margin_rate

    def run(self, price_data: dict, calendar: pd.DatetimeIndex):
        assets = list(price_data.keys())
        n_days = len(calendar)
        # positions in contracts/shares (integer)
        positions = {a: 0 for a in assets}
        trades = []
        cash = float(self.initial_capital)
        equity_series = []

        # Precompute closures for quick index mapping
        closes = {a: price_data[a]["close"].values for a in assets}
        vols = {a: price_data[a]["volume"].values for a in assets}
        df_by_asset = {a: price_data[a] for a in assets}

        for i, date in enumerate(calendar):
            # build signals
            signals = {}
            for a in assets:
                sig = momentum_breakdown_signal(df_by_asset[a], i)
                signals[a] = sig

            # Determine allocation: distribute a fixed fraction of capital across signaled assets
            signaled = [a for a, s in signals.items() if s == 1]
            target_positions = {a: 0 for a in assets}
            if signaled:
                # total short allocation = 50% of capital (aggressive for crisis test)
                total_short_notional = 0.50 * self.initial_capital
                per_asset_notional = total_short_notional / len(signaled)
                for a in signaled:
                    price = closes[a][i]
                    if price <= 0:
                        units = 0
                    else:
                        units = int(per_asset_notional / price)
                    target_positions[a] = -units  # negative for short

            # Execute position changes with slippage; update cash
            for a in assets:
                cur = positions[a]
                tgt = target_positions[a]
                if cur != tgt:
                    price = closes[a][i]
                    # if entering a larger short (more negative), assume worse execution for seller: receive slightly less
                    if tgt < cur:
                        exec_price = price * (1 - self.slippage)
                    else:
                        # covering / reducing short: you pay a little more
                        exec_price = price * (1 + self.slippage)
                    # Cash changes: (old_pos - new_pos) * exec_price
                    # Example: short 0 -> -10 contracts: cash += (0 - (-10)) * exec_price => +10*exec_price (you receive proceeds)
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

            # Mark-to-market equity = cash + sum(position * price)
            mtm = cash
            for a in assets:
                mtm += positions[a] * float(closes[a][i])
            # Deduct margin requirement (maintenance) as unrealized margin hold (not a cash subtraction but reported)
            # For simplicity we'll ensure equity >= required margin else record a margin call and force cover largest short
            required_margin = sum([abs(positions[a]) * float(closes[a][i]) * self.margin_rate for a in assets])
            if mtm < required_margin:
                # margin shortfall: force reduce shorts (cover 50% of notional from largest position)
                # find largest absolute dollar short
                shorts = [(a, abs(positions[a]) * float(closes[a][i])) for a in assets if positions[a] < 0]
                if shorts:
                    shorts.sort(key=lambda x: x[1], reverse=True)
                    a_to_cover, _ = shorts[0]
                    cur = positions[a_to_cover]
                    cover_units = int(abs(cur) * 0.5)  # cover half
                    if cover_units > 0:
                        exec_price = float(closes[a_to_cover][i]) * (1 + self.slippage)  # covering is costly
                        new_pos = cur + cover_units  # e.g., -100 + 50 => -50
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
                # record margin call event in trades as a trade with note (we keep it simple)

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
        # simple annualized return approx over scenario days
        days = (equity_series.index[-1] - equity_series.index[0]).days or 1
        annualized = ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (365.0 / days) - 1) if equity_series.iloc[0] > 0 else 0.0
        return {"start_equity": float(equity_series.iloc[0]), "end_equity": float(equity_series.iloc[-1]),
                "net_pnl": net_pnl, "max_drawdown": max_dd, "annualized_return": annualized}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide", page_title="GFC (Lehman) Short-only Backtest")
st.title("Lehman / GFC Short-only Backtest — one-file")

with st.sidebar:
    st.header("Lehman (GFC) controls")
    st.markdown(f"**Scenario:** {SCENARIO['label']}")
    st.write(f"Date range: {SCENARIO['start']} → {SCENARIO['end']}")
    sectors = st.multiselect("Sectors (short-only proxies)", options=list(ASSET_MAP.keys()),
                             default=["Financials", "Real Estate", "Energy"])
    initial_cap = st.number_input("Initial capital (USD)", value=1_000_000, step=100_000)
    slippage_bps = st.number_input("Slippage (bps)", value=50)  # larger for crisis
    margin_rate = st.number_input("Margin rate (fraction)", value=0.12)
    run_bt = st.button("Run Lehman / GFC backtest")

st.markdown("**Notes:** This script uses synthetic placeholder price series for quick local testing. For production replace data loader with real CME / vendor data feeds. Short-only; allocation = 50% initial capital distributed equally among signaled assets.")

if run_bt:
    st.info("Generating data and running backtest for GFC scenario...")
    # map selected sectors to assets
    selected_assets = []
    for s in sectors:
        selected_assets += ASSET_MAP.get(s, [])
    if not selected_assets:
        st.error("Please select at least one sector.")
    else:
        start = SCENARIO["start"]
        end = SCENARIO["end"]
        price_data, calendar = load_price_matrix(selected_assets, start, end)
        engine = SimpleShortBacktest(initial_capital=initial_cap, slippage_bps=slippage_bps, margin_rate=margin_rate)
        result = engine.run(price_data, calendar)

        # Show metrics
        st.subheader("Backtest summary")
        metrics = result["metrics"]
        st.json(metrics)

        # Equity chart
        st.subheader("Equity curve (MTM)")
        eq_df = result["equity"].reset_index()
        eq_df = eq_df.rename(columns={"index": "date"}) if "index" in eq_df.columns else eq_df
        eq_df = eq_df.set_index("date")
        st.line_chart(eq_df["equity"])

        # trades table
        st.subheader("Trades / Executions")
        if not result["trades"].empty:
            st.dataframe(result["trades"].sort_values("date"))
        else:
            st.write("No trades recorded.")

        # Save YAML doc following your documentation checklist
        run_doc = {
            "run_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "preset": SCENARIO["preset"],
            "scenario": SCENARIO["label"],
            "date_range": {"start": SCENARIO["start"], "end": SCENARIO["end"]},
            "strategy": "momentum_breakdown_short_v1",
            "params": {"initial_capital": initial_cap, "slippage_bps": slippage_bps, "margin_rate": margin_rate,
                       "sectors": sectors, "assets": selected_assets},
            "notes": "Scenario label: GFC (Lehman). Short-only stress test; synthetic prices."
        }
        yml_name = f"{datetime.utcnow().strftime('%Y%m%d')}_{SCENARIO['tag']}_momentum_breakdown.yml"
        yml_path = DOCS / yml_name
        with open(yml_path, "w") as f:
            yaml.dump(run_doc, f)
        st.success(f"Saved run doc: {yml_path}")

        # Export a one-page PDF placeholder
        if st.button("Export one-page PDF (summary)"):
            pdf_path = DOCS / f"{datetime.utcnow().strftime('%Y%m%d')}_{SCENARIO['tag']}_summary.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 760, "Backtest summary — Lehman / GFC (Short-only)")
            c.setFont("Helvetica", 10)
            c.drawString(50, 740, f"Run date: {run_doc['run_date']}")
            c.drawString(50, 725, f"Scenario: {run_doc['scenario']}")
            c.drawString(50, 710, f"Date range: {SCENARIO['start']} → {SCENARIO['end']}")
            c.drawString(50, 695, f"Initial capital: ${initial_cap:,.0f}")
            c.drawString(50, 680, f"Net PnL: {metrics.get('net_pnl'):.2f}")
            c.drawString(50, 665, f"Max Drawdown: {metrics.get('max_drawdown'):.4f}")
            c.drawString(50, 650, f"Annualized (approx): {metrics.get('annualized_return'):.2%}")
            c.drawString(50, 630, "Top trades (first 10):")
            trades = result["trades"].sort_values("date")
            y = 615
            for _, r in trades.head(10).iterrows():
                line = f"{pd.to_datetime(r['date']).date()} {r['asset']} {int(r['from_pos'])}->{int(r['to_pos'])} @ {r['exec_price']:.2f}"
                c.drawString(60, y, line)
                y -= 12
                if y < 50:
                    break
            c.save()
            st.success(f"Exported PDF: {pdf_path}")

st.markdown("---")
st.markdown("If you want this swapped into a multi-file repo, or want real CME data wired in next, tell me which connector (Quandl/CME S3 / your vendor) and I will produce that next.")