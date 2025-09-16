# backtest_gfc_onefile_with_morning_model.py
"""
All-in-one Streamlit backtest for the Lehman / GFC scenario (short-only)
with TraderMorningModel prerequisite gating.

Dependencies:
  pip install streamlit pandas numpy pyyaml pyarrow reportlab
Run:
  streamlit run backtest_gfc_onefile_with_morning_model.py
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
# Add the TraderMorningModel framework (user provided)
# ---------------------------
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import datetime as _dt
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class MarketSnapshot:
    timestamp: _dt.datetime
    futures: Dict[str, float]                    # e.g. {"ES": -0.8, "NQ": -1.2}  -> percent moves
    etf_prices: Dict[str, float]                 # e.g. {"IYR": 80.2, "VNQ": 68.4}
    etf_volumes: Dict[str, float]                # raw volume or relative-to-average
    credit_rating_changes: Dict[str, str]        # e.g. {"RMBS": "unchanged"}
    default_rates: Dict[str, float]              # e.g. {"mortgage_30d": 0.045} (fractions or rel-to-avg)
    financials: Dict[str, float]                 # e.g. {"XLF": -2.1}
    treasuries: Dict[str, float]                 # e.g. {"2y": 0.75, "10y": 1.8} yields
    volatility: Dict[str, float]                 # e.g. {"VIX": 22.3}
    headlines: List[str]                         # short list of important headlines

@dataclass
class RoutineConfig:
    etf_volume_threshold: float = 2.0            # e.g. volume >= 2x normal is 'high'
    default_rate_spike_threshold: float = 1.5    # e.g. 1.5x normal -> spike
    discretionary_risk_pct: float = 0.01         # max risk per trade if trading (1% of capital)
    capital: float = 1_000_000                   # example default account capital
    debug: bool = True

@dataclass
class TradeRecommendation:
    instrument: str
    action: str                                   # e.g. 'buy_put', 'short_etf', 'buy_treasury'
    size_usd: float
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)

class TraderMorningModel:
    def __init__(self, config: RoutineConfig):
        self.config = config
        self.state: Dict[str, Any] = {}
        self.log = logging.getLogger("TraderMorningModel")
        if config.debug:
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.INFO)

    # ---------- Core routine steps ----------
    def boot_systems(self):
        self.log.info("Booting systems: market feeds, OMS, execution, comms.")
        self.state['systems_ok'] = True

    def quick_body_reset(self):
        # Placeholder for human routine (coffee, breathing)
        self.state['ready'] = True
        self.log.debug("Human: coffee & breathing done.")

    def fetch_market_snapshot(self) -> MarketSnapshot:
        raise NotImplementedError("Connect this to your market data feeds")

    def read_headlines(self, snapshot: MarketSnapshot) -> List[str]:
        important = [h for h in snapshot.headlines if len(h) > 10][:5]
        self.state['headlines'] = important
        self.log.debug(f"Headlines triaged: {important}")
        return important

    def risk_and_pnl_snapshot(self) -> Dict[str, Any]:
        pnl = {"overnight_pnl": 0.0, "available_margin": self.config.capital * 0.2}
        self.state['pnl_snapshot'] = pnl
        self.log.debug(f"P&L snapshot: {pnl}")
        return pnl

    def reconcile_positions(self):
        self.state['positions_ok'] = True
        self.log.debug("Positions reconciled and stale orders cleaned.")

    # ---------- Decision logic (the "model" part) ----------
    def detect_discrepancy(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        result = {"flag": False, "score": 0.0, "reasons": []}

        mort_default = snapshot.default_rates.get("mortgage_rel_to_avg", None)
        if mort_default is not None and mort_default >= self.config.default_rate_spike_threshold:
            result['score'] += 1.0
            result['reasons'].append("mortgage_default_spike")

        unchanged_count = sum(1 for v in snapshot.credit_rating_changes.values() if v.lower() in ("unchanged", "no change"))
        if unchanged_count >= 1 and mort_default and mort_default >= self.config.default_rate_spike_threshold:
            result['score'] += 1.0
            result['reasons'].append("ratings_lagging")

        high_vol_etfs = [etf for etf, vol in snapshot.etf_volumes.items() if vol >= self.config.etf_volume_threshold]
        if high_vol_etfs:
            result['score'] += 1.0
            result['reasons'].append("etf_high_volume")
            result['etf_list'] = high_vol_etfs

        if result['score'] >= 2.0:
            result['flag'] = True

        self.log.debug(f"Discrepancy detection: {result}")
        return result


def make_recommendations(self, snapshot: MarketSnapshot, discrepancy: Dict[str,Any]) -> List[TradeRecommendation]:
        recs: List[TradeRecommendation] = []
        capital = self.config.capital
        risk_per_trade = capital * self.config.discretionary_risk_pct

        if discrepancy.get("flag"):
            reasons = discrepancy.get("reasons", [])
            self.log.info(f"ALERT: discrepancy flagged for reasons {reasons}")

            recs.append(TradeRecommendation(
                instrument="TLT", action="buy", size_usd=risk_per_trade,
                reason="flight-to-quality hedge (credit stress detected)"
            ))

            for etf in discrepancy.get("etf_list", [])[:3]:
                recs.append(TradeRecommendation(
                    instrument=etf, action="buy_put_spread", size_usd=risk_per_trade*0.5,
                    reason=f"cheap convexity on {etf} due to volume+default mismatch",
                    params={"expiry_weeks": 4, "width_pct": 3}
                ))

            if snapshot.financials.get("XLF", 0.0) < -1.0:
                recs.append(TradeRecommendation(
                    instrument="XLF", action="buy_puts", size_usd=risk_per_trade*0.75,
                    reason="protect banking exposure as financials underperform"
                ))

            for etf in discrepancy.get("etf_list", [])[:2]:
                recs.append(TradeRecommendation(
                    instrument=etf, action="short_etf", size_usd=risk_per_trade*0.5,
                    reason="fade heavy-volume 'distribution' into uncertainty",
                    params={"scale_in_steps": 3}
                ))
        else:
            self.log.info("No discrepancy flag; return conservative watchlist actions.")
            recs.append(TradeRecommendation(
                instrument="IYR", action="watch", size_usd=0.0,
                reason="Monitor real-estate ETF for signs of distribution"
            ))

        self.state['recommendations'] = recs
        self.log.debug(f"Recommendations: {recs}")
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

        self.log.info("Routine complete. Ready for market open.")
        self.state['last_summary'] = summary
        return summary

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