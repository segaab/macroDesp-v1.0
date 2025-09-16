# backtest_gfc_onefile_with_morning_model.py
"""
All-in-one Streamlit backtest for the Lehman / GFC scenario (short-only)
with TraderMorningModel prerequisite gating.

Dependencies:
  pip install streamlit pandas numpy pyyaml pyarrow reportlab yahooquery
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

# attempt to import yahooquery (optional). If missing, we'll fallback to synthetic.
try:
    from yahooquery import Ticker
    HAS_YAHOOQUERY = True
except Exception:
    Ticker = None
    HAS_YAHOOQUERY = False

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
        # Stub: reconcile blotter <-> custody
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
                reason="flight_to-quality hedge (credit stress detected)"
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
# Utilities: data generation / yahoofetch
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

def fetch_price_data_yahoo(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Try to fetch daily OHLCV from yahooquery for `symbol` between start and end (inclusive).
    Returns DataFrame with columns ['date','open','high','low','close','volume'] or None on failure.
    """
    if not HAS_YAHOOQUERY:
        return None
    try:
        t = Ticker(symbol)
        hist = t.history(start=start, end=end, interval="1d")
        if hist is None or len(hist) == 0:
            return None
        # yahooquery.history may return MultiIndex when multiple symbols requested;
        # handle both single and multi symbol returns.
        if isinstance(hist.index, pd.MultiIndex):
            # pick the symbol's subset
            if symbol in hist.index.get_level_values(0):
                df = hist.xs(symbol, level=0)
            else:
                # if the symbol isn't present, return None
                return None
        else:
            df = hist.copy()
        # normalize and ensure columns exist
        df = df.reset_index()
        # some fields are 'adjclose' or 'close' depending on feed; prefer 'adjclose' if present
        if 'adjclose' in df.columns:
            df = df.rename(columns={'adjclose': 'close'})
        # ensure required cols
        expected = ['date','open','high','low','close','volume']
        missing = [c for c in expected if c not in df.columns]
        if missing:
            # try best-effort mapping, otherwise fail
            return None
        df = df[['date','open','high','low','close','volume']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception:
        return None

def load_price_matrix(selected_assets, start, end):
    """
    For each asset attempt to fetch via Yahoo; on failure generate synthetic and cache.
    Returns dict asset -> DataFrame and calendar index.
    """
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    calendar = pd.date_range(start=start_ts, end=end_ts, freq="B")
    price_data = {}
    for asset in selected_assets:
        # attempt yahoo fetch
        df = fetch_price_data_yahoo(asset, start, end)
        if df is None:
            # fallback to synthetic generator (keeps prior behavior)
            df = generate_synthetic(asset, start_ts, end_ts)
        # reindex to full calendar and forward-fill / backfill to avoid holes
        df = df.set_index('date').sort_index()
        df = df.reindex(calendar).ffill().bfill().reset_index().rename(columns={'index': 'date'})
        price_data[asset] = df
    return price_data, calendar

# ---------------------------
# Config / Scenario (GFC) — start set to one day BEFORE Lehman bankruptcy (2008-09-14)
# ---------------------------
SCENARIO = {
    "preset": "GFC",
    "label": "GFC (Lehman bankruptcy - Sep 15, 2008 → Mar 31, 2009)",
    "start": "2008-09-14",
    "end": "2009-03-31",
    "tag": "gfc2008"
}

# Sector -> asset proxies
ASSET_MAP = {
    "Energy": ["CL"],                   # crude futures proxy (try symbol as-is; adjust to CL=F if you prefer)
    "Financials": ["XLF"],              # financial ETF proxy
    "Industrials": ["XLI"],
    "Consumer Discretionary": ["XLY"],
    "Materials": ["XLB"],
    "Real Estate": ["XLRE"],
    "Technology": ["NQ"]                # Nasdaq futures proxy (may be NQ=F on Yahoo)
}

# Output paths
ROOT = Path(".")
DATA_ROOT = ROOT / "data_placeholder"
DOCS = ROOT / "docs"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Strategy: momentum_breakdown_signal
# ---------------------------
def momentum_breakdown_signal(df: pd.DataFrame, idx: int) -> int:
    if idx < 50:
        return 0
    window = df.iloc[max(0, idx-100):idx+1]
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
# Backtest engine (supports allocation scaling)
# ---------------------------
class SimpleShortBacktest:
    def __init__(self, initial_capital=1_000_000, slippage_bps=30, margin_rate=0.1, allocation_pct=0.50):
        """
        allocation_pct: fraction of initial capital used for total short notional (default 0.50)
        """
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
            signals = {}
            for a in assets:
                sig = momentum_breakdown_signal(df_by_asset[a], i)
                signals[a] = sig

            signaled = [a for a, s in signals.items() if s == 1]
            target_positions = {a: 0 for a in assets}
            if signaled:
                total_short_notional = self.allocation_pct * self.initial_capital
                per_asset_notional = total_short_notional / len(signaled)
                for a in signaled:
                    price = closes[a][i]
                    if price <= 0:
                        units = 0
                    else:
                        units = int(per_asset_notional / price)
                    target_positions[a] = -units

            for a in assets:
                cur = positions[a]
                tgt = target_positions[a]
                if cur != tgt:
                    price = closes[a][i]
                    if tgt < cur:
                        exec_price = price * (1 - self.slippage)
                    else:
                        exec_price = price * (1 + self.slippage)
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
st.set_page_config(layout="wide", page_title="GFC (Lehman) Short-only Backtest with Morning Model")
st.title("Lehman / GFC Short-only Backtest — Morning Model Prereq")

with st.sidebar:
    st.header("Lehman (GFC) controls")
    st.markdown(f"**Scenario:** {SCENARIO['label']}")
    st.write(f"Date range: {SCENARIO['start']} → {SCENARIO['end']}")
    sectors = st.multiselect("Sectors (short-only proxies)", options=list(ASSET_MAP.keys()),
                             default=["Financials", "Real Estate", "Energy"])
    initial_cap = st.number_input("Initial capital (USD)", value=1_000_000, step=100_000)
    slippage_bps = st.number_input("Slippage (bps)", value=50)  # larger for crisis
    margin_rate = st.number_input("Margin rate (fraction)", value=0.12)
    # Morning model params
    etf_vol_threshold = st.number_input("ETF volume threshold (x avg)", value=2.0)
    default_rate_threshold = st.number_input("Default rate spike threshold (x avg)", value=1.6)
    discretionary_risk_pct = st.number_input("Discretionary risk % (per rec)", value=0.01)
    run_bt = st.button("Run Lehman / GFC backtest (with prerequisite)")

st.markdown("**Notes:** The TraderMorningModel runs on a snapshot created from the synthetic or Yahoo data. If it flags a discrepancy, the backtest will *scale down* allocation automatically (50% → 25%) as the gating rule.")

if run_bt:
    st.info("Generating data, running morning model, then backtest...")

    selected_assets = []
    for s in sectors:
        selected_assets += ASSET_MAP.get(s, [])
    if not selected_assets:
        st.error("Please select at least one sector.")
    else:
        start = SCENARIO["start"]
        end = SCENARIO["end"]
        price_data, calendar = load_price_matrix(selected_assets, start, end)

        # ---------------------------
        # Build MarketSnapshot from latest data (simple heuristics)
        # ---------------------------
        latest_idx = -1
        # futures percent moves: compute pct change from prev day for each asset
        futures_pct = {}
        etf_prices = {}
        etf_vols_rel = {}
        financials_map = {}
        for a in selected_assets:
            df = price_data[a]
            if len(df) >= 2:
                today = df.iloc[-1]
                prev = df.iloc[-2]
                pct = 100.0 * (float(today["close"]) - float(prev["close"])) / float(prev["close"]) if prev["close"] != 0 else 0.0
            else:
                pct = 0.0
            futures_pct[a] = round(pct, 3)
            etf_prices[a] = float(df.iloc[-1]["close"])
            # compute vol_rel = today_vol / 50-day avg vol
            vol_today = float(df.iloc[-1]["volume"])
            vol_avg50 = float(df["volume"].rolling(50, min_periods=1).mean().iloc[-1])
            vol_rel = vol_today / (vol_avg50 + 1e-9)
            etf_vols_rel[a] = round(vol_rel, 3)
            # financials mapping: if XLF present
            if a == "XLF":
                financials_map["XLF"] = round(pct, 3)

        # simple default proxy: if XLF or XLRE dropped sharply recently, mark default proxy high
        avg_recent_pct = np.mean([v for v in futures_pct.values()])
        # heuristic: if avg drop worse than -2% -> default_rel ~1.8 else 1.0
        mortgage_rel_to_avg = 1.8 if avg_recent_pct <= -2.0 else 1.2 if avg_recent_pct <= -1.0 else 1.0

        # treasuries and volatility proxies (placeholders derived from data)
        # VIX proxy: use cross-asset realized vol (std of returns over last 20 business days) * sqrt(252)
        vol_proxy = {}
        all_rets = []
        for a in selected_assets:
            series = price_data[a]["close"].pct_change().dropna().tail(20)
            if len(series) > 1:
                rv = series.std() * np.sqrt(252) * 100
            else:
                rv = 20.0
            vol_proxy[a] = round(rv, 2)
            all_rets.extend(series.tolist())
        vix_proxy = round(np.std(all_rets) * np.sqrt(252) * 100, 2) if all_rets else 20.0

        treasuries = {"2y": 0.50, "10y": 1.80}
        snapshot = MarketSnapshot(
            timestamp=pd.to_datetime(end),
            futures=futures_pct,
            etf_prices=etf_prices,
            etf_volumes=etf_vols_rel,
            credit_rating_changes={"RMBS": "unchanged"},
            default_rates={"mortgage_rel_to_avg": mortgage_rel_to_avg},
            financials=financials_map,
            treasuries=treasuries,
            volatility={"VIX": vix_proxy},
            headlines=[
                "Lehman files for bankruptcy protection",
                "Credit markets seize up",
                "Fed announces emergency liquidity facilities",
                "Bank funding stress intensifies",
            ]
        )

        # ---------------------------
        # Run morning model
        # ---------------------------
        cfg = RoutineConfig(
            etf_volume_threshold=etf_vol_threshold,
            default_rate_spike_threshold=default_rate_threshold,
            discretionary_risk_pct=discretionary_risk_pct,
            capital=initial_cap,
            debug=True
        )
        morning_model = TraderMorningModel(cfg)
        summary = morning_model.run_morning_routine(snapshot)

        st.subheader("Morning Model Summary")
        st.json(summary)

        discrepancy_flag = summary["discrepancy"]["flag"]
        alloc_pct = 0.25 if discrepancy_flag else 0.50
        st.write(f"**Backtest allocation scaling:** {alloc_pct*100:.0f}% of capital")

        # ---------------------------
        # Run backtest
        # ---------------------------
        engine = SimpleShortBacktest(
            initial_capital=initial_cap,
            slippage_bps=slippage_bps,
            margin_rate=margin_rate,
            allocation_pct=alloc_pct
        )
        results = engine.run(price_data, calendar)

        st.subheader("Backtest Metrics")
        st.json(results["metrics"])

        st.subheader("Equity Curve")
        st.line_chart(results["equity"][["equity", "required_margin"]])

        st.subheader("Trades")
        st.dataframe(results["trades"])

        # ---------------------------
        # Export YAML + placeholder PDF
        # ---------------------------
        yaml_doc = {
            "run_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "preset": SCENARIO["preset"],
            "scenario": SCENARIO["label"],
            "date_range": {"start": SCENARIO["start"], "end": SCENARIO["end"]},
            "strategy": "momentum_breakdown_short_v1",
            "notes": f"Scenario label: {SCENARIO['label']}. Morning model flag={discrepancy_flag}."
        }
        yml_path = DOCS / f"{datetime.utcnow().strftime('%Y%m%d')}__{SCENARIO['tag']}__momentum.yml"
        with open(yml_path, "w") as f:
            yaml.safe_dump(yaml_doc, f)
        st.success(f"Saved YAML config to {yml_path}")

        pdf_path = DOCS / f"{datetime.utcnow().strftime('%Y%m%d')}__{SCENARIO['tag']}__momentum.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(100, 750, "Backtest Report")
        c.drawString(100, 730, f"Scenario: {SCENARIO['label']}")
        c.drawString(100, 710, f"Metrics: {results['metrics']}")
        c.save()
        st.success(f"Saved placeholder PDF report to {pdf_path}")