# backtest_gfc_onefile_with_morning_model.py
"""
All-in-one Streamlit backtest for the Lehman / GFC scenario (short-only)
with TraderMorningModel prerequisite gating.

Dependencies:
  pip install streamlit pandas numpy pyarrow reportlab
Run:
  streamlit run backtest_gfc_onefile_with_morning_model.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------------------
# Minimal dataclass shim (no imports)
# ---------------------------
def dataclass(cls):
    annotations = getattr(cls, "__annotations__", {})
    def __init__(self, **kwargs):
        for k, _ in annotations.items():
            setattr(self, k, kwargs.get(k))
    cls.__init__ = __init__
    return cls

# ---------------------------
# TraderMorningModel framework
# ---------------------------
from typing import Dict, List, Any

@dataclass
class MarketSnapshot:
    timestamp: datetime
    futures: Dict[str, float]
    etf_prices: Dict[str, float]
    etf_volumes: Dict[str, float]
    credit_rating_changes: Dict[str, str]
    default_rates: Dict[str, float]
    financials: Dict[str, float]
    treasuries: Dict[str, float]
    volatility: Dict[str, float]
    headlines: List[str]

@dataclass
class RoutineConfig:
    etf_volume_threshold: float = 2.0
    default_rate_spike_threshold: float = 1.5
    discretionary_risk_pct: float = 0.01
    capital: float = 1_000_000
    debug: bool = True

@dataclass
class TradeRecommendation:
    instrument: str
    action: str
    size_usd: float
    reason: str
    params: Dict[str, Any] = None

# ---------------------------
# HF Inference client helper
# ---------------------------
import os
from huggingface_hub import InferenceClient

def classify_headlines_finbert(client, headlines, neg_threshold=0.6):
    """
    Returns True if sentiment negative above threshold, False otherwise
    """
    if not headlines:
        return False
    for h in headlines:
        resp = client.text_classification(h, model="ProsusAI/finbert")
        if isinstance(resp, list) and resp:
            best = max(resp, key=lambda x: x.get("score", 0.0))
            if best.get("label","") == "negative" and best.get("score",0.0) >= neg_threshold:
                return True
    return False

# ---------------------------
# TraderMorningModel
# ---------------------------
import logging

class TraderMorningModel:
    def __init__(self, config: RoutineConfig, hf_client):
        self.config = config
        self.state: Dict[str, Any] = {}
        self.log = logging.getLogger("TraderMorningModel")
        self.hf_client = hf_client
        self.log.setLevel(logging.DEBUG if config.debug else logging.INFO)

    def boot_systems(self):
        self.log.info("Booting systems...")
        self.state['systems_ok'] = True

    def quick_body_reset(self):
        self.state['ready'] = True
        self.log.debug("Human reset done.")

    def read_headlines(self, snapshot: MarketSnapshot) -> List[str]:
        important = [h for h in snapshot.headlines if len(h) > 10][:5]
        self.state['headlines'] = important
        self.log.debug(f"Headlines triaged: {important}")
        return important

    def risk_and_pnl_snapshot(self) -> Dict[str, Any]:
        pnl = {"overnight_pnl": 0.0, "available_margin": self.config.capital*0.2}
        self.state['pnl_snapshot'] = pnl
        return pnl

    def reconcile_positions(self):
        self.state['positions_ok'] = True

    def detect_discrepancy(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        result = {"flag": False, "score": 0.0, "reasons": [], "etf_list":[]}

        # Check headline sentiment
        headlines = snapshot.headlines
        negative_sentiment = classify_headlines_finbert(self.hf_client, headlines, neg_threshold=0.6)
        if negative_sentiment:
            result['score'] += 1.0
            result['reasons'].append("negative_headlines")

        mort_default = snapshot.default_rates.get("mortgage_rel_to_avg", None)
        if mort_default and mort_default >= self.config.default_rate_spike_threshold:
            result['score'] += 1.0
            result['reasons'].append("mortgage_default_spike")

        high_vol_etfs = [etf for etf, vol in snapshot.etf_volumes.items() if vol >= self.config.etf_volume_threshold]
        if high_vol_etfs:
            result['score'] += 1.0
            result['reasons'].append("etf_high_volume")
            result['etf_list'] = high_vol_etfs

        if result['score'] >= 2.0:
            result['flag'] = True

        return result

    def make_recommendations(self, snapshot: MarketSnapshot, discrepancy: Dict[str,Any]) -> List[TradeRecommendation]:
        recs: List[TradeRecommendation] = []
        capital = self.config.capital
        risk_per_trade = capital * self.config.discretionary_risk_pct

        if discrepancy.get("flag"):
            recs.append(TradeRecommendation("TLT","buy",risk_per_trade,"flight-to-quality hedge"))
            for etf in discrepancy.get("etf_list", [])[:3]:
                recs.append(TradeRecommendation(etf,"buy_put_spread",risk_per_trade*0.5,f"convexity on {etf}"))

        else:
            recs.append(TradeRecommendation("IYR","watch",0.0,"Monitor real-estate ETF"))

        self.state['recommendations'] = recs
        return recs

    def run_morning_routine(self, snapshot: MarketSnapshot) -> Dict[str,Any]:
        self.boot_systems()
        self.quick_body_reset()
        headlines = self.read_headlines(snapshot)
        pnl = self.risk_and_pnl_snapshot()
        self.reconcile_positions()
        discrepancy = self.detect_discrepancy(snapshot)
        recommendations = self.make_recommendations(snapshot, discrepancy)

        summary = {
            "timestamp": snapshot.timestamp.isoformat(),
            "headlines": headlines,
            "pnl_snapshot": pnl,
            "discrepancy": discrepancy,
            "recommendations": [r.__dict__ for r in recommendations]
        }
        self.state['last_summary'] = summary
        return summary

# ---------------------------
# Helpers for synthetic data, news scraping, and ritual
# ---------------------------
def generate_synthetic(asset, start, end):
    path = Path(".")/f"{asset}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df[(df["date"]>=start)&(df["date"]<=end)].copy()
    dates = pd.date_range(start=start,end=end,freq="B")
    n = len(dates)
    rng = np.random.RandomState(sum(bytearray(asset.encode()))%2**32)
    mu = -0.0008 if asset in ("XLF","XLRE") else -0.0003
    sigma = 0.03
    returns = rng.normal(mu,sigma,n)
    price = 100*np.exp(np.cumsum(returns))
    vol = rng.randint(500,5000,n)
    df = pd.DataFrame({"date":dates,"close":price,"volume":vol})
    df.to_parquet(path)
    return df

# ---------------------------
# Backtest engine
# ---------------------------
class SimpleShortBacktest:
    def __init__(self, initial_capital=1_000_000, slippage_bps=30, margin_rate=0.1, allocation_pct=0.5):
        self.initial_capital = initial_capital
        self.slippage = slippage_bps / 10000.0
        self.margin_rate = margin_rate
        self.allocation_pct = allocation_pct

    def run(self, price_data, calendar):
        assets = list(price_data.keys())
        positions = {a: 0 for a in assets}
        trades = []
        cash = float(self.initial_capital)
        equity_series = []

        closes = {a: price_data[a]["close"].values for a in assets}
        df_by_asset = {a: price_data[a] for a in assets}

        for i, date in enumerate(calendar):
            signals = {
                a: 1
                if i >= 50 and df_by_asset[a]["close"].iloc[i] < df_by_asset[a]["close"].rolling(50).mean().iloc[i]
                else 0
                for a in assets
            }
            signaled = [a for a, s in signals.items() if s == 1]
            target_positions = {a: 0 for a in assets}
            if signaled:
                total_short = self.allocation_pct * self.initial_capital
                per_asset = total_short / len(signaled)
                for a in signaled:
                    price = closes[a][i]
                    units = int(per_asset / price) if price > 0 else 0
                    target_positions[a] = -units

            for a in assets:
                cur = positions[a]
                tgt = target_positions[a]
                if cur != tgt:
                    price = closes[a][i]
                    exec_price = price * (1 - self.slippage if tgt < cur else 1 + self.slippage)
                    cash += (cur - tgt) * exec_price
                    positions[a] = tgt
                    trades.append(
                        {"date": date, "asset": a, "from_pos": cur, "to_pos": tgt, "exec_price": exec_price, "units": tgt - cur}
                    )

            mtm = cash + sum([positions[a] * closes[a][i] for a in assets])
            required_margin = sum([abs(positions[a]) * closes[a][i] * self.margin_rate for a in assets])
            equity_series.append({"date": date, "equity": mtm, "required_margin": required_margin})

        equity_df = pd.DataFrame(equity_series).set_index("date")
        return equity_df, pd.DataFrame(trades)

# ---------------------------
# Ritual runner
# ---------------------------
def run_morning_ritual_for_period(morning_model, price_snapshots, start_date, end_date):
    ritual_summaries = []
    current_date = start_date
    while current_date <= end_date:
        snapshot = price_snapshots.get(current_date, None)
        if snapshot:
            summary = morning_model.run_morning_routine(snapshot)
            ritual_summaries.append(summary)
        current_date += pd.Timedelta(days=1)
    return ritual_summaries

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide", page_title="GFC Short-only Backtest")
st.title("Lehman / GFC Short-only Backtest — Morning Model")

with st.sidebar:
    st.header("Controls")
    sectors = st.multiselect("Sectors", options=["Financials", "Real Estate", "Energy"], default=["Financials", "Real Estate"])
    initial_cap = st.number_input("Initial capital", value=1_000_000)
    run_bt = st.button("Run Backtest")

# ---------------------------
# Prepare synthetic price data
# ---------------------------
start_date = pd.Timestamp("2008-09-10")  # 3 days before Lehman collapse
end_date = pd.Timestamp("2008-09-15")    # Collapse date
calendar = pd.date_range(start=start_date, end=end_date, freq="B")

price_data = {}
for sector in sectors:
    asset = {"Financials": "XLF", "Real Estate": "XLRE", "Energy": "XLE"}.get(sector, "XLF")
    df = generate_synthetic(asset, start=start_date, end=end_date)
    price_data[asset] = df

# ---------------------------
# Initialize HF inference client
# ---------------------------
hf_token = os.getenv("HF_TOKEN", "")
hf_client = InferenceClient(provider="hf-inference", api_key=hf_token)

# ---------------------------
# Initialize Morning Model
# ---------------------------
routine_config = RoutineConfig(capital=initial_cap)
morning_model = TraderMorningModel(routine_config, hf_client)

# ---------------------------
# Run morning ritual
# ---------------------------
price_snapshots = {}
for d in calendar:
    snapshot = MarketSnapshot(
        timestamp=d,
        futures={a: price_data[a].loc[price_data[a]["date"] == d, "close"].values[0] for a in price_data},
        etf_prices={a: price_data[a].loc[price_data[a]["date"] == d, "close"].values[0] for a in price_data},
        etf_volumes={a: price_data[a].loc[price_data[a]["date"] == d, "volume"].values[0] for a in price_data},
        credit_rating_changes={},
        default_rates={"mortgage_rel_to_avg": 1.6 if d == pd.Timestamp("2008-09-12") else 1.0},
        financials={},
        treasuries={},
        volatility={},
        headlines=[
            "Markets tumble amid banking sector fears",
            "Credit default swaps spike in early September",
            "Lehman bankruptcy looming"
        ] if d <= pd.Timestamp("2008-09-12") else []
    )
    price_snapshots[d] = snapshot

ritual_start = start_date
ritual_end = end_date
ritual_results = run_morning_ritual_for_period(morning_model, price_snapshots, ritual_start, ritual_end)

# ---------------------------
# Run short-only backtest
# ---------------------------
backtester = SimpleShortBacktest(initial_capital=initial_cap)
equity_df, trades_df = backtester.run(price_data, calendar)

# ---------------------------
# Display results
# ---------------------------
st.subheader("Morning Ritual Summaries")
for s in ritual_results:
    st.json(s)

st.subheader("Backtest Equity Curve")
st.line_chart(equity_df["equity"])

st.subheader("Executed Trades")
st.dataframe(trades_df)