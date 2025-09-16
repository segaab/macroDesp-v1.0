# chunk1_core_and_morning_model.py
"""
Chunk 1: Core dataclasses, HF helpers, and TraderMorningModel with ritual support.
Drop this at the top of your single-file script (preserves prior code).
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import datetime as _dt
from datetime import datetime, timedelta
import json
import requests

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)

# ---------------------------
# Dataclasses
# ---------------------------
@dataclass
class MarketSnapshot:
    timestamp: _dt.datetime
    futures: Dict[str, float]                    # e.g. {"ES": -0.8, "NQ": -1.2}  -> percent moves
    etf_prices: Dict[str, float]                 # e.g. {"IYR": 80.2, "VNQ": 68.4}
    etf_volumes: Dict[str, float]                # raw volume or relative-to-average
    credit_rating_changes: Dict[str, str]        # e.g. {"RMBS": "unchanged"}
    default_rates: Dict[str, float]              # e.g. {"mortgage_rel_to_avg": 1.8}
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
# HF helpers (InferenceClient optional / requests fallback)
# ---------------------------
# try huggingface_hub.InferenceClient first (if installed). If not, we'll use direct HTTP requests.
try:
    from huggingface_hub import InferenceClient  # type: ignore
    HAS_HF_CLIENT = True
except Exception:
    InferenceClient = None
    HAS_HF_CLIENT = False

def init_hf_client_from_token(token: Optional[str]) -> Optional[Any]:
    """
    Return a huggingface_hub.InferenceClient if available and token provided, else None.
    """
    if not token:
        return None
    if HAS_HF_CLIENT:
        try:
            client = InferenceClient(provider="hf-inference", api_key=token)
            return client
        except Exception as e:
            logging.warning(f"Failed to init InferenceClient: {e}")
            return None
    return None

def hf_classify_via_requests(token: str, text: str, model: str = "ProsusAI/finbert") -> Optional[List[Dict[str, Any]]]:
    """
    Use Hugging Face Inference HTTP endpoint as a fallback when InferenceClient isn't available.
    Returns the parsed JSON (usually a list of label/score dicts) or None on failure.
    """
    if not token:
        return None
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": text}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            logging.debug(f"HF HTTP classify failed: {resp.status_code} {resp.text[:200]}")
            return None
    except Exception as e:
        logging.debug(f"HF HTTP classify exception: {e}")
        return None

def classify_headlines_finbert(hf_client: Optional[Any], hf_token: Optional[str], headlines: List[str], neg_threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    Try HF InferenceClient -> requests fallback -> keyword heuristic.
    Returns list of dicts: {"headline": str, "label": "NEGATIVE"/"POSITIVE"/"NEUTRAL", "score": float}
    """
    results: List[Dict[str, Any]] = []
    for h in headlines:
        labeled = None
        if hf_client is not None:
            try:
                resp = hf_client.text_classification(h, model="ProsusAI/finbert")
                if isinstance(resp, list) and len(resp) > 0:
                    best = max(resp, key=lambda x: x.get("score", 0.0))
                    labeled = {"headline": h, "label": best.get("label", "").upper(), "score": float(best.get("score", 0.0))}
            except Exception as e:
                logging.debug(f"HF client call failed, will fallback: {e}")
                labeled = None

        if labeled is None and hf_token:
            resp = hf_classify_via_requests(hf_token, h, model="ProsusAI/finbert")
            if isinstance(resp, list) and len(resp) > 0:
                best = max(resp, key=lambda x: x.get("score", 0.0))
                # HF HTTP often returns label strings that may be lowercase or mixed; normalize
                label = best.get("label", "")
                try:
                    score = float(best.get("score", 0.0))
                except Exception:
                    score = 0.0
                labeled = {"headline": h, "label": label.upper(), "score": score}

        if labeled is None:
            # Keyword heuristic fallback
            lower = h.lower()
            neg_score = 0.7 if any(k in lower for k in ["bankrupt", "collapse", "panic", "plunge", "meltdown", "default", "fear"]) else 0.1
            lbl = "NEGATIVE" if neg_score >= neg_threshold else "NEUTRAL"
            labeled = {"headline": h, "label": lbl, "score": neg_score}
        results.append(labeled)
    return results

# ---------------------------
# Curated Lehman headlines (date-aware)
# ---------------------------
def get_lehman_headlines_with_dates() -> List[Dict[str, str]]:
    """
    Curated headlines with approximate dates. Used for historical snapshots.
    """
    return [
        {"date": "2008-09-12", "headline": "Markets jitter ahead of key bank meetings"},  # synthetic filler
        {"date": "2008-09-13", "headline": "Banks seek capital as tensions rise"},       # synthetic filler
        {"date": "2008-09-14", "headline": "Banking crisis: Lehman Brothers files for bankruptcy protection"},
        {"date": "2008-09-15", "headline": "Investors fear domino effect after Lehman Brothers folds"},
        {"date": "2008-09-15", "headline": "Pain Continues on Wall Street as Lehman Goes Bankrupt"},
    ]

# ---------------------------
# TraderMorningModel (with ritual running over multiple days)
# ---------------------------
class TraderMorningModel:
    def __init__(self, config: RoutineConfig, hf_client: Optional[Any] = None, hf_token: Optional[str] = None):
        self.config = config
        self.state: Dict[str, Any] = {}
        self.hf_client = hf_client
        self.hf_token = hf_token
        self.log = logging.getLogger("TraderMorningModel")
        self.log.setLevel(logging.DEBUG if config.debug else logging.INFO)

    # basic routines (unchanged)
    def boot_systems(self):
        self.log.info("Booting systems")
        self.state['systems_ok'] = True

    def quick_body_reset(self):
        self.state['ready'] = True

    def read_headlines(self, snapshot: MarketSnapshot) -> List[str]:
        important = [h for h in snapshot.headlines if len(h) > 5][:5]
        self.state['headlines'] = important
        return important

    def risk_and_pnl_snapshot(self) -> Dict[str, Any]:
        pnl = {"overnight_pnl": 0.0, "available_margin": self.config.capital * 0.2}
        self.state['pnl_snapshot'] = pnl
        return pnl

    def reconcile_positions(self):
        self.state['positions_ok'] = True

    def detect_discrepancy(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Combines default spike, rating lag, ETF volume, and negative headline signal.
        """
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

        # headline sentiment
        headlines = self.read_headlines(snapshot)
        sentiment = classify_headlines_finbert(self.hf_client, self.hf_token, headlines, neg_threshold=0.6)
        result['headline_sentiment'] = sentiment
        neg_count = sum(1 for s in sentiment if s.get("label", "").upper().startswith("NEG"))
        if neg_count >= 1:
            result['score'] += 1.0
            result['reasons'].append("headline_negative")
            result['neg_headline_count'] = neg_count

        if result['score'] >= 2.0:
            result['flag'] = True

        return result

    def make_recommendations(self, snapshot: MarketSnapshot, discrepancy: Dict[str, Any]) -> List[TradeRecommendation]:
        recs: List[TradeRecommendation] = []
        capital = self.config.capital
        risk_per_trade = capital * self.config.discretionary_risk_pct

        if discrepancy.get("flag"):
            recs.append(TradeRecommendation("TLT", "buy", risk_per_trade, "flight-to-quality hedge"))
            for etf in discrepancy.get("etf_list", [])[:3]:
                recs.append(TradeRecommendation(etf, "buy_put_spread", risk_per_trade * 0.5,
                                                f"cheap convexity on {etf}", params={"expiry_weeks": 4, "width_pct": 3}))
            if snapshot.financials.get("XLF", 0.0) < -1.0:
                recs.append(TradeRecommendation("XLF", "buy_puts", risk_per_trade * 0.75, "protect banking exposure"))
            for etf in discrepancy.get("etf_list", [])[:2]:
                recs.append(TradeRecommendation(etf, "short_etf", risk_per_trade * 0.5,
                                                "fade heavy-volume distribution", params={"scale_in_steps": 3}))
        else:
            recs.append(TradeRecommendation("IYR", "watch", 0.0, "Monitor real-estate ETF"))
        self.state['recommendations'] = recs
        return recs

    def run_morning_routine(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Single-day run (keeps original behavior).
        """
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

    def run_morning_ritual(self, build_snapshot_fn, start_date: _dt.datetime, end_date: _dt.datetime) -> Dict[str, Any]:
        """
        Run the morning routine across multiple consecutive days (ritual).
        - build_snapshot_fn(date) -> MarketSnapshot
        Returns aggregated results: per-day summaries and an overall flag (if any day flagged).
        """
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        per_day = {}
        any_flag = False
        max_score = 0.0

        while current <= end:
            try:
                snapshot = build_snapshot_fn(current)
            except Exception as e:
                logging.debug(f"build_snapshot_fn failed for {current}: {e}")
                current += pd.Timedelta(days=1)
                continue

            summary = self.run_morning_routine(snapshot)
            per_day[str(current.date())] = summary
            score = summary["discrepancy"].get("score", 0.0)
            if score > max_score:
                max_score = score
            if summary["discrepancy"].get("flag", False):
                any_flag = True
            current += pd.Timedelta(days=1)

        return {"per_day": per_day, "any_flag": any_flag, "max_score": max_score}


# chunk2_ui_and_backtest.py
"""
Chunk 2: scenario, data generation, backtest engine and Streamlit UI.
This chunk assumes chunk1_core_and_morning_model.py is present above in the same file.
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------
# Scenario & mapping (preserved)
# ---------------------------
SCENARIO = {
    "preset": "GFC",
    "label": "GFC (Lehman bankruptcy - Sep 15, 2008 → Mar 31, 2009)",
    "start": "2008-09-14",
    "end": "2009-03-31",
    "tag": "gfc2008"
}
CRASH_DATE = pd.to_datetime("2008-09-15")
RITUAL_START = CRASH_DATE - pd.Timedelta(days=3)  # three days before crash (2008-09-12)

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
# Synthetic data / loader (preserved)
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
    mu = -0.0008 if asset in ("XLF", "XLRE") else -0.0003
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
# Momentum signal (preserved)
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
# Backtest engine (preserved)
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
                        mtm = cash + sum([positions[a] * float(closes[a][i]) for a in assets])

            equity_series.append({"date": pd.to_datetime(date), "equity": mtm, "required_margin": required_margin})

        equity_df = pd.DataFrame(equity_series).set_index("date")
        trades_df = pd.DataFrame(trades)
        metrics = self._compute_metrics(equity_df["equity"])
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
# Streamlit UI (ritual runs 3 days before crash)
# ---------------------------
st.set_page_config(layout="wide", page_title="GFC Short-only Backtest with Morning Ritual")
st.title("Lehman / GFC Short-only Backtest — Morning Ritual + FinBERT")

with st.sidebar:
    st.header("Lehman (GFC) controls")
    st.markdown(f"**Scenario:** {SCENARIO['label']}")
    st.write(f"Scenario range: {SCENARIO['start']} → {SCENARIO['end']}")
    sectors = st.multiselect("Sectors (short-only proxies)", options=list(ASSET_MAP.keys()),
                             default=["Financials", "Real Estate", "Energy"])
    initial_cap = st.number_input("Initial capital (USD)", value=1_000_000, step=100_000)
    slippage_bps = st.number_input("Slippage (bps)", value=50)
    margin_rate = st.number_input("Margin rate (fraction)", value=0.12)
    etf_vol_threshold = st.number_input("ETF volume threshold (x avg)", value=2.0)
    default_rate_threshold = st.number_input("Default rate spike threshold (x avg)", value=1.6)
    discretionary_risk_pct = st.number_input("Discretionary risk % (per rec)", value=0.01)
    hf_token_input = st.text_input("Hugging Face API Key (optional)", type="password")
    run_bt = st.button("Run Lehman / GFC backtest (with 3-day ritual)")

if run_bt:
    st.info("Generating data, running 3-day morning ritual, then backtest...")

    selected_assets = [a for s in sectors for a in ASSET_MAP.get(s, [])]
    if not selected_assets:
        st.error("Please select at least one sector.")
    else:
        # load price matrix (preserved behaviour)
        price_data, calendar = load_price_matrix(selected_assets, SCENARIO["start"], SCENARIO["end"])

        # helper to build snapshot for a given date (uses price_data)
        def build_snapshot_for_date(date: pd.Timestamp) -> MarketSnapshot:
            # find nearest business date in calendar
            date = pd.to_datetime(date)
            if date not in calendar:
                # snap to nearest previous business day
                prev_dates = calendar[calendar <= date]
                if len(prev_dates) == 0:
                    idx = 0
                else:
                    idx = len(prev_dates) - 1
            else:
                idx = list(calendar).index(date)
            futures_pct = {}
            etf_prices = {}
            etf_vols_rel = {}
            financials_map = {}
            for a in selected_assets:
                df = price_data[a]
                row = df.iloc[idx]
                prev_row = df.iloc[max(0, idx-1)]
                pct = 100.0 * (float(row["close"]) - float(prev_row["close"])) / float(prev_row["close"]) if prev_row["close"] != 0 else 0.0
                futures_pct[a] = round(pct, 3)
                etf_prices[a] = float(row["close"])
                vol_today = float(row["volume"])
                vol_avg50 = float(df["volume"].rolling(50, min_periods=1).mean().iloc[idx])
                etf_vols_rel[a] = round(vol_today / (vol_avg50 + 1e-9), 3)
                if a == "XLF":
                    financials_map["XLF"] = round(pct, 3)

            avg_recent_pct = np.mean([v for v in futures_pct.values()]) if futures_pct else 0.0
            mortgage_rel_to_avg = 1.8 if avg_recent_pct <= -2.0 else 1.2 if avg_recent_pct <= -1.0 else 1.0
            vol_proxy = {}
            all_rets = []
            for a in selected_assets:
                series = price_data[a]["close"].pct_change().dropna().tail(20)
                rv = series.std() * np.sqrt(252) * 100 if len(series) > 1 else 20.0
                vol_proxy[a] = round(rv, 2)
                all_rets.extend(series.tolist())
            vix_proxy = round(np.std(all_rets) * np.sqrt(252) * 100, 2) if all_rets else 20.0
            treasuries = {"2y": 0.50, "10y": 1.5}

            # pick headlines for this date from curated set (fall back to generic if missing)
            curated = get_lehman_headlines_with_dates()
            date_str = date.strftime("%Y-%m-%d")
            todays = [c["headline"] for c in curated if c["date"] == date_str]
            if not todays:
                # fallback: use the closest earlier headlines (or curated last)
                earlier = [c["headline"] for c in curated if c["date"] <= date_str]
                todays = earlier[-3:] if earlier else [c["headline"] for c in curated[:3]]

            snapshot = MarketSnapshot(
                timestamp=pd.Timestamp(date),
                futures=futures_pct,
                etf_prices=etf_prices,
                etf_volumes=etf_vols_rel,
                credit_rating_changes={"RMBS": "unchanged"},
                default_rates={"mortgage_rel_to_avg": mortgage_rel_to_avg},
                financials=financials_map,
                treasuries=treasuries,
                volatility={"VIX": vix_proxy},
                headlines=todays
            )
            return snapshot

        # init morning model with HF client/token
        hf_client = init_hf_client_from_token(hf_token_input) if hf_token_input else None
        morning_cfg = RoutineConfig(
            etf_volume_threshold=etf_vol_threshold,
            default_rate_spike_threshold=default_rate_threshold,
            discretionary_risk_pct=discretionary_risk_pct,
            capital=initial_cap,
            debug=True
        )
        morning_model = TraderMorningModel(morning_cfg, hf_client=hf_client, hf_token=hf_token_input if hf_token_input else None)

        # run ritual from RITUAL_START to day-before-crash (i.e., CRASH_DATE - 1 day)
        ritual_start = RITUAL_START
        ritual_end = CRASH_DATE - pd.Timedelta(days=1)
        ritual_results = morning_model.run_morning_ritual(build_snapshot_for_date, ritual_start, ritual_end)

        st.subheader("Morning Ritual Results (per day)")
        st.json(ritual_results["per_day"])
        st.write(f"Any day flagged: {ritual_results['any_flag']}  — max_score: {ritual_results['max_score']}")

        # decide allocation (scale down if any_flag True)
        allocation_pct = 0.25 if ritual_results["any_flag"] else 0.50
        if ritual_results["any_flag"]:
            st.warning("Morning ritual flagged stress in the run-up → scaling allocation to 25%")
        else:
            st.success("No major ritual flags → using standard allocation (50%)")

        # run backtest using allocation_pct
        bt = SimpleShortBacktest(initial_capital=initial_cap, slippage_bps=slippage_bps, margin_rate=margin_rate, allocation_pct=allocation_pct)
        bt_res = bt.run(price_data, calendar)

        st.subheader("Equity Curve")
        st.line_chart(bt_res["equity"]["equity"])

        st.subheader("Backtest Metrics")
        st.json(bt_res["metrics"])

        st.subheader("Trades (sample)")
        st.dataframe(bt_res["trades"].head(30))

        # Save YAML doc (preserved style)
        import yaml
        yml = {
            "run_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "preset": SCENARIO["preset"],
            "scenario": SCENARIO["label"],
            "date_range": {"start": SCENARIO["start"], "end": SCENARIO["end"]},
            "strategy": "momentum_breakdown_short_v1",
            "notes": f"Ritual run {ritual_start.date()} → {ritual_end.date()}, any_flag={ritual_results['any_flag']}"
        }
        yml_path = DOCS / f"{datetime.utcnow().strftime('%Y%m%d')}__{SCENARIO['tag']}__momentum.yml"
        with open(yml_path, "w") as f:
            yaml.safe_dump(yml, f)
        st.success(f"Saved YAML to {yml_path}")