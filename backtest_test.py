# backtest_gfc_onefile_with_morning_model.py
"""
All-in-one Streamlit backtest for the Lehman / GFC scenario (short-only)
with TraderMorningModel prerequisite gating.

Dependencies:
  pip install streamlit pandas numpy pyyaml pyarrow reportlab huggingface_hub
Run:
  streamlit run backtest_gfc_onefile_with_morning_model.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import requests
from typing import Dict, List, Any

# ---------------------------
# TraderMorningModel helpers (already include dataclasses)
# ---------------------------
import logging
logging.basicConfig(level=logging.INFO)

@dataclass
class MarketSnapshot:
    timestamp: datetime
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
    params: Dict[str, Any] = field(default_factory=dict)

# ---------------------------
# Hugging Face FinBERT scoring helper
# ---------------------------
def classify_headlines_finbert(client, hf_token, headlines: List[str], neg_threshold=0.6) -> List[Dict[str,Any]]:
    labeled_headlines = []
    for h in headlines:
        payload = {"inputs": h}
        headers = {"Authorization": f"Bearer {hf_token}"}
        resp = []
        try:
            r = requests.post(
                "https://api-inference.huggingface.co/models/ProsusAI/finbert",
                headers=headers, json=payload, timeout=10
            )
            resp = r.json()
        except Exception as e:
            print(f"HF inference error: {e}")
            resp = []

        # filter only valid dicts
        valid_items = [r for r in resp if isinstance(r, dict) and "score" in r and "label" in r]
        if valid_items:
            best = max(valid_items, key=lambda x: x["score"])
            labeled = {"headline": h, "label": best["label"].upper(), "score": float(best["score"])}
        else:
            lower = h.lower()
            neg_score = 0.7 if any(k in lower for k in ["bankrupt", "collapse", "panic", "plunge", "meltdown", "default", "fear"]) else 0.1
            lbl = "NEGATIVE" if neg_score >= neg_threshold else "NEUTRAL"
            labeled = {"headline": h, "label": lbl, "score": neg_score}
        labeled_headlines.append(labeled)
    return labeled_headlines

# ---------------------------
# TraderMorningModel
# ---------------------------
class TraderMorningModel:
    def __init__(self, config: RoutineConfig, hf_token:str):
        self.config = config
        self.state: Dict[str, Any] = {}
        self.hf_token = hf_token
        self.log = logging.getLogger("TraderMorningModel")
        if config.debug:
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.INFO)

    def boot_systems(self):
        self.log.info("Booting systems")
        self.state['systems_ok'] = True

    def quick_body_reset(self):
        self.state['ready'] = True
        self.log.debug("Human: coffee & breathing done.")

    def fetch_market_snapshot(self) -> MarketSnapshot:
        raise NotImplementedError("Connect to live or synthetic market data")

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
        self.log.debug("Positions reconciled.")

    # --------------------------
    # Decision logic
    # --------------------------
    def detect_discrepancy(self, snapshot: MarketSnapshot) -> Dict[str,Any]:
        result = {"flag": False, "score": 0.0, "reasons": []}
        headlines = self.read_headlines(snapshot)
        sentiment = classify_headlines_finbert(self, self.hf_token, headlines, neg_threshold=0.6)
        negative_count = sum(1 for s in sentiment if s["label"]=="NEGATIVE" and s["score"]>=0.6)

        mort_default = snapshot.default_rates.get("mortgage_rel_to_avg", None)
        if mort_default and mort_default >= self.config.default_rate_spike_threshold:
            result['score'] += 1.0
            result['reasons'].append("mortgage_default_spike")

        if negative_count >= 2:
            result['score'] += 1.0
            result['reasons'].append("negative_headlines")

        high_vol_etfs = [etf for etf, vol in snapshot.etf_volumes.items() if vol >= self.config.etf_volume_threshold]
        if high_vol_etfs:
            result['score'] += 1.0
            result['reasons'].append("etf_high_volume")
            result['etf_list'] = high_vol_etfs

        if result['score'] >= 2.0:
            result['flag'] = True
        self.log.debug(f"Discrepancy detection: {result}")
        return result


# ---------------------------
# TraderMorningModel continued
# ---------------------------
    def run_morning_routine(self, snapshot: MarketSnapshot) -> Dict[str,Any]:
        self.boot_systems()
        self.quick_body_reset()
        self.reconcile_positions()
        pnl = self.risk_and_pnl_snapshot()
        discrepancy = self.detect_discrepancy(snapshot)
        summary = {
            "pnl": pnl,
            "discrepancy": discrepancy,
            "timestamp": snapshot.timestamp,
        }
        return summary

    def run_morning_ritual(self, build_snapshot_func, start_date: datetime, end_date: datetime) -> List[Dict[str,Any]]:
        ritual_results = []
        current_date = start_date
        while current_date <= end_date:
            snapshot = build_snapshot_func(current_date)
            summary = self.run_morning_routine(snapshot)
            ritual_results.append(summary)
            current_date += timedelta(days=1)
        return ritual_results

# ---------------------------
# Synthetic market snapshot builder (for Lehman scenario)
# ---------------------------
def build_snapshot_for_date(date: datetime) -> MarketSnapshot:
    # Simulate synthetic futures & ETFs with negative skew near crash
    np.random.seed(int(date.strftime("%Y%m%d")))
    futures = {"ES": np.random.normal(-1.5, 0.5), "NQ": np.random.normal(-1.8,0.6)}
    etf_prices = {"IYR": 80.0 + np.random.normal(0,1), "VNQ": 68.0 + np.random.normal(0,1)}
    etf_volumes = {"IYR": np.random.uniform(2.0,5.0), "VNQ": np.random.uniform(1.0,4.0)}
    credit_rating_changes = {"RMBS": "downgrade"} if date >= datetime(2008,9,12) else {"RMBS": "unchanged"}
    default_rates = {"mortgage_rel_to_avg": 1.6 if date >= datetime(2008,9,12) else 0.8}
    financials = {"XLF": np.random.normal(-3.0,0.5)}
    treasuries = {"2y": 0.75, "10y": 1.8}
    volatility = {"VIX": 30 + np.random.normal(0,2)}
    headlines = [
        "Lehman Brothers faces bankruptcy filing",
        "Markets in panic after financial collapse",
        "Investors fear systemic contagion",
        "Treasury steps in to stabilize credit markets",
        "Major banks halt lending amid liquidity crunch"
    ]
    return MarketSnapshot(
        timestamp=date,
        futures=futures,
        etf_prices=etf_prices,
        etf_volumes=etf_volumes,
        credit_rating_changes=credit_rating_changes,
        default_rates=default_rates,
        financials=financials,
        treasuries=treasuries,
        volatility=volatility,
        headlines=headlines
    )

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="GFC Backtest Dashboard", layout="wide")
st.title("Macro Despair Backtest - Short-Only Strategies (GFC)")

hf_token = st.text_input("HuggingFace HF_TOKEN", type="password")
ritual_start = st.date_input("Morning ritual start date", datetime(2008,9,8))
ritual_end = st.date_input("Morning ritual end date", datetime(2008,9,10))

if st.button("Run Morning Ritual"):
    if not hf_token:
        st.error("HF_TOKEN required for FinBERT sentiment analysis")
    else:
        morning_model = TraderMorningModel(RoutineConfig(), hf_token)
        ritual_results = morning_model.run_morning_ritual(
            build_snapshot_for_date,
            datetime.combine(ritual_start, datetime.min.time()),
            datetime.combine(ritual_end, datetime.min.time())
        )
        st.success("Morning ritual complete")
        for res in ritual_results:
            st.write(res)

# ---------------------------
# Additional analysis / plotting
# ---------------------------
st.subheader("Synthetic Market Snapshot Example")
snapshot = build_snapshot_for_date(datetime(2008,9,15))
st.write(snapshot)

# Example ETF volatility plot
etf_vol_df = pd.DataFrame(snapshot.etf_volumes.items(), columns=["ETF","RelVol"])
st.bar_chart(etf_vol_df.set_index("ETF"))

st.subheader("Headline Sentiment Analysis")
headlines_labeled = classify_headlines_finbert(morning_model, hf_token, snapshot.headlines)
st.table(pd.DataFrame(headlines_labeled))

# ---------------------------
# End of script
# ---------------------------