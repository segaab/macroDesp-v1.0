# ---------------------------
# Imports & Setup
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from huggingface_hub import InferenceClient
import os

# ---------------------------
# Hugging Face Client Setup
# ---------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing Hugging Face API key. Please set HF_TOKEN in your environment.")

hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

# ---------------------------
# Data Classes
# ---------------------------
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class MarketSnapshot:
    date: pd.Timestamp
    asset_prices: Dict[str, float]
    asset_volumes: Dict[str, float]
    headline_scores: Dict[str, float]

# ---------------------------
# Scenario Config (Lehman Brothers collapse)
# ---------------------------
CRASH_DATE = datetime(2008, 9, 15)
RITUAL_START = CRASH_DATE - timedelta(days=3)   # begin 3 days before collapse
RITUAL_END = CRASH_DATE

# ---------------------------
# Helper Functions
# ---------------------------
def classify_headlines_finbert(client: InferenceClient, headlines: List[str], neg_threshold: float = 0.6):
    """Classify news headlines using FinBERT and return those with strong negative sentiment."""
    results = []
    for h in headlines:
        try:
            resp = client.text_classification(h, model="ProsusAI/finbert")
            if resp:
                best = max(resp, key=lambda x: x.get("score", 0.0))
                results.append(best)
        except Exception as e:
            print(f"HF Inference failed for headline: {h}, error: {e}")
            continue

    return [
        r for r in results
        if r.get("label") == "negative" and r.get("score", 0.0) >= neg_threshold
    ]


# ---------------------------
# Morning Model
# ---------------------------
class MorningModel:
    def __init__(self, hf_client: InferenceClient):
        self.hf_client = hf_client

    def detect_discrepancy(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        headlines = list(snapshot.headline_scores.keys())
        if not headlines:
            return {"negative_sentiment": []}

        negative_sentiment = classify_headlines_finbert(
            self.hf_client, headlines, neg_threshold=0.6
        )
        return {"negative_sentiment": negative_sentiment}

    def run_morning_routine(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        discrepancy = self.detect_discrepancy(snapshot)
        return {
            "date": snapshot.date,
            "discrepancy": discrepancy,
            "asset_prices": snapshot.asset_prices,
        }

# ---------------------------
# Ritual Loop
# ---------------------------
def run_morning_ritual_for_period(
    model: MorningModel, snapshots: List[MarketSnapshot],
    start: datetime, end: datetime
):
    results = []
    for snap in snapshots:
        if start <= snap.date <= end:
            summary = model.run_morning_routine(snap)
            results.append(summary)
    return results

# ---------------------------
# Mock Snapshot Builder
# (replace this with YahooQuery or your own data source)
# ---------------------------
def build_snapshot_for_date(date: datetime) -> MarketSnapshot:
    mock_prices = {"XLF": 20.0, "CL": 100.0}
    mock_volumes = {"XLF": 1_000_000, "CL": 500_000}
    mock_headlines = {
        "Lehman struggles to find buyer": -0.7,
        "Markets face stress over credit fears": -0.8,
    }
    return MarketSnapshot(
        date=date,
        asset_prices=mock_prices,
        asset_volumes=mock_volumes,
        headline_scores=mock_headlines,
    )

# ---------------------------
# Build Ritual Results
# ---------------------------
price_snapshots = [
    build_snapshot_for_date(RITUAL_START + timedelta(days=i))
    for i in range((RITUAL_END - RITUAL_START).days + 1)
]

morning_model = MorningModel(hf_client)
ritual_results = run_morning_ritual_for_period(
    morning_model, price_snapshots, RITUAL_START, RITUAL_END
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Lehman Brothers Collapse Backtest Dashboard")
st.subheader("Morning Ritual Analysis (3 days before → collapse)")

for res in ritual_results:
    st.markdown(f"**Date:** {res['date'].strftime('%Y-%m-%d')}")
    st.write("Asset Prices:", res["asset_prices"])
    st.write("Negative Sentiment:", res["discrepancy"]["negative_sentiment"])
    st.markdown("---")