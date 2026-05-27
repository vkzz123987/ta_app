"""
Technical Analysis Streamlit App
================================

What it does
------------
- Asks the user for a stock / ETF / index / FX / crypto / futures ticker.
- Converts common human ticker formats into Yahoo Finance / yfinance symbols.
- Pulls OHLCV data from yfinance.
- Produces three separate reports: daily, weekly, and 15-minute.
- Draws candlestick + volume charts with moving averages, support/resistance,
  and basic pattern annotations.
- Generates a rule-based technical analysis paragraph.
- Optionally generate prompt for users to copy/paste to chatbots for more analysis.
- Optionally sends compact market data to an LLM endpoint and displays the LLM's
  analysis instead of, or alongside, the local report.

Run
---
pip install -r requirements.txt
streamlit run technical_analysis_app.py

Important
---------
This app is for research and education only. It is not financial advice.
Verify data, signals, and execution risks before making any trading decision.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# -----------------------------
# Configuration
# -----------------------------

TIMEFRAMES: Dict[str, Dict[str, Any]] = {
    "Daily": {
        "period": "18mo",
        "interval": "1d",
        "pivot_order": 3,
        "lookback_bars_for_levels": 260,
        "chart_title_suffix": "18M / 1D",
    },
    "Weekly": {
        "period": "5y",
        "interval": "1wk",
        "pivot_order": 2,
        "lookback_bars_for_levels": 220,
        "chart_title_suffix": "5Y / 1W",
    },
    "15-Minute": {
        # Yahoo intraday history is limited. 30d is usually a safe default for 15m.
        "period": "30d",
        "interval": "15m",
        "pivot_order": 6,
        "lookback_bars_for_levels": 500,
        "chart_title_suffix": "30D / 15m",
    },
}

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# -----------------------------
# Data models
# -----------------------------

@dataclass
class Level:
    price: float
    touches: int
    distance_pct: float


@dataclass
class LevelSet:
    supports: List[Level]
    resistances: List[Level]
    all_levels: List[Level]


@dataclass
class Signal:
    action: str
    score: int
    bias: str
    reasons: List[str]
    risk_note: str


# -----------------------------
# Ticker normalization
# -----------------------------

EXCHANGE_SUFFIX_MAP = {
    "HK": ".HK",
    "HKG": ".HK",
    "ASX": ".AX",
    "AU": ".AX",
    "AX": ".AX",
    "TSX": ".TO",
    "TO": ".TO",
    "TSE": ".TO",  # Ambiguous globally, but common for Toronto in North America.
    "TSXV": ".V",
    "V": ".V",
    "LSE": ".L",
    "LON": ".L",
    "UK": ".L",
    "JP": ".T",
    "TYO": ".T",
    "T": ".T",
    "SH": ".SS",
    "SHA": ".SS",
    "SS": ".SS",
    "SZ": ".SZ",
    "SHE": ".SZ",
    "KS": ".KS",
    "KQ": ".KQ",
    "KR": ".KS",
    "SW": ".SW",
    "SIX": ".SW",
    "SI": ".SI",
    "SG": ".SI",
}

US_EXCHANGES = {"US", "NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}

SPECIAL_TICKERS = {
    "SP500": "^GSPC",
    "S&P500": "^GSPC",
    "S&P 500": "^GSPC",
    "GSPC": "^GSPC",
    "DOW": "^DJI",
    "DJI": "^DJI",
    "NASDAQ": "^IXIC",
    "IXIC": "^IXIC",
    "NDX": "^NDX",
    "RUSSELL2000": "^RUT",
    "RUT": "^RUT",
    "VIX": "^VIX",
    "HSI": "^HSI",
    "NIKKEI": "^N225",
    "N225": "^N225",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "WTI": "CL=F",
    "OIL": "CL=F",
    "BRENT": "BZ=F",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}


def normalize_ticker(user_input: str) -> str:
    """Convert common human ticker formats into yfinance-friendly symbols.

    Examples:
    - "aapl" -> "AAPL"
    - "AAPL US" -> "AAPL"
    - "2899 HK" -> "2899.HK"
    - "CBA AU" -> "CBA.AX"
    - "RY TSX" -> "RY.TO"
    - "EUR/USD" -> "EURUSD=X"
    - "BTCUSD" -> "BTC-USD"
    - "SP500" -> "^GSPC"

    This cannot perfectly map every global exchange symbol. The app validates
    candidates against yfinance and tells the user which symbol was used.
    """
    raw = user_input.strip().upper()
    raw = raw.replace("，", ",")
    raw = re.sub(r"\s+", " ", raw)

    if raw in SPECIAL_TICKERS:
        return SPECIAL_TICKERS[raw]

    # Bloomberg-like or TradingView-like format: EXCHANGE:TICKER.
    if ":" in raw:
        exchange, symbol = raw.split(":", 1)
        exchange = exchange.strip().upper()
        symbol = symbol.strip().upper().replace(".", "-")
        if exchange in US_EXCHANGES:
            return symbol
        if exchange in EXCHANGE_SUFFIX_MAP:
            return f"{symbol}{EXCHANGE_SUFFIX_MAP[exchange]}"

    # FX pairs: EUR/USD, EURUSD, USDJPY, etc.
    fx_match = re.fullmatch(r"([A-Z]{3})[\-/ ]?([A-Z]{3})", raw)
    if fx_match:
        base, quote = fx_match.groups()
        if base in {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF", "CNY", "HKD"} and quote in {
            "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF", "CNY", "HKD"
        }:
            return f"{base}{quote}=X"

    # Crypto pairs: BTCUSD, ETHUSD, BTC-USD.
    crypto_match = re.fullmatch(r"(BTC|ETH|SOL|BNB|XRP|ADA|DOGE|AVAX|DOT|LINK)[\-/ ]?(USD|USDT)", raw)
    if crypto_match:
        coin, quote = crypto_match.groups()
        # Yahoo crypto uses USD pairs most reliably.
        return f"{coin}-USD"

    # Space-separated market suffix: "2899 HK", "CBA AU", "RY TSX".
    parts = raw.split(" ")
    if len(parts) == 2:
        symbol, market = parts
        symbol = symbol.replace(".", "-")  # US class shares: BRK.B -> BRK-B if typed as BRK B.
        if market in US_EXCHANGES:
            return symbol
        if market in EXCHANGE_SUFFIX_MAP:
            return f"{symbol}{EXCHANGE_SUFFIX_MAP[market]}"

    # Already has a yfinance suffix or special Yahoo markers.
    if any(raw.endswith(suffix) for suffix in EXCHANGE_SUFFIX_MAP.values()) or raw.startswith("^") or raw.endswith("=F") or raw.endswith("=X"):
        return raw

    # US class shares are often BRK-B in Yahoo, while many users type BRK.B.
    if re.fullmatch(r"[A-Z]{1,5}\.[A-Z]", raw):
        return raw.replace(".", "-")

    return raw


def candidate_tickers(user_input: str) -> List[str]:
    normalized = normalize_ticker(user_input)
    raw = user_input.strip().upper()
    candidates = [normalized, raw]

    # Sometimes users enter class shares with a dot.
    if "." in raw and not any(raw.endswith(suffix) for suffix in EXCHANGE_SUFFIX_MAP.values()):
        candidates.append(raw.replace(".", "-"))

    # Deduplicate while preserving order.
    seen = set()
    result = []
    for item in candidates:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


@st.cache_data(ttl=60 * 15, show_spinner=False)
def quick_validate_ticker(candidate: str) -> bool:
    try:
        data = yf.download(
            candidate,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return data is not None and not data.empty
    except Exception:
        return False


def resolve_ticker(user_input: str) -> str:
    for candidate in candidate_tickers(user_input):
        if quick_validate_ticker(candidate):
            return candidate
    # Return best-effort normalized ticker so the main fetch can show a useful error.
    return normalize_ticker(user_input)


# -----------------------------
# Market data and indicators
# -----------------------------

@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_market_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}, period={period}, interval={interval}.")

    # yfinance can return a MultiIndex depending on version and parameters.
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = [col[0] for col in df.columns]

    df = df.rename(columns={c: str(c).title() for c in df.columns})

    # Some instruments do not have true volume. Keep Volume as 0 so charts still render.
    if "Volume" not in df.columns:
        df["Volume"] = 0

    missing = [col for col in ["Open", "High", "Low", "Close"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns from yfinance data: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Datetime"
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDHist"] = df["MACD"] - df["MACDSignal"]
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["VolumeMA20"] = df["Volume"].rolling(20).mean()
    df["BBMid"] = df["SMA20"]
    bb_std = df["Close"].rolling(20).std()
    df["BBUpper"] = df["BBMid"] + 2 * bb_std
    df["BBLower"] = df["BBMid"] - 2 * bb_std
    return df


# -----------------------------
# Support / resistance and patterns
# -----------------------------

def find_pivots(df: pd.DataFrame, order: int = 3) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
    highs: List[Tuple[pd.Timestamp, float]] = []
    lows: List[Tuple[pd.Timestamp, float]] = []
    if len(df) < order * 2 + 1:
        return highs, lows

    high_values = df["High"].values
    low_values = df["Low"].values
    idx = df.index

    for i in range(order, len(df) - order):
        high_window = high_values[i - order : i + order + 1]
        low_window = low_values[i - order : i + order + 1]
        current_high = high_values[i]
        current_low = low_values[i]

        if current_high == np.nanmax(high_window) and current_high > np.nanmedian(high_window):
            highs.append((idx[i], float(current_high)))
        if current_low == np.nanmin(low_window) and current_low < np.nanmedian(low_window):
            lows.append((idx[i], float(current_low)))

    return highs, lows


def cluster_levels(prices: List[float], current_price: float, latest_atr: float, max_levels: int = 8) -> List[Level]:
    clean_prices = sorted([float(p) for p in prices if p and np.isfinite(p)])
    if not clean_prices:
        return []

    tolerance = max(current_price * 0.004, latest_atr * 0.35 if np.isfinite(latest_atr) else 0)
    clusters: List[List[float]] = []

    for price in clean_prices:
        if not clusters:
            clusters.append([price])
            continue
        cluster_avg = float(np.mean(clusters[-1]))
        if abs(price - cluster_avg) <= tolerance:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    levels = []
    for cluster in clusters:
        level_price = float(np.mean(cluster))
        touches = len(cluster)
        distance_pct = (level_price - current_price) / current_price * 100
        levels.append(Level(price=level_price, touches=touches, distance_pct=distance_pct))

    # Prefer frequently touched and nearby levels.
    levels.sort(key=lambda x: (-x.touches, abs(x.distance_pct)))
    return levels[:max_levels]


def calculate_levels(df: pd.DataFrame, order: int, lookback_bars: int) -> LevelSet:
    recent = df.tail(min(len(df), lookback_bars)).copy()
    current_price = float(recent["Close"].iloc[-1])
    latest_atr = float(recent["ATR14"].dropna().iloc[-1]) if recent["ATR14"].notna().any() else current_price * 0.02

    pivot_highs, pivot_lows = find_pivots(recent, order=order)
    prices = [p for _, p in pivot_highs + pivot_lows]

    # Force recent range extremes into the candidate list.
    for window in [20, 50, 100]:
        if len(recent) >= window:
            prices.append(float(recent["High"].tail(window).max()))
            prices.append(float(recent["Low"].tail(window).min()))

    all_levels = cluster_levels(prices, current_price, latest_atr, max_levels=12)
    supports = sorted([lvl for lvl in all_levels if lvl.price < current_price], key=lambda x: abs(x.distance_pct))[:4]
    resistances = sorted([lvl for lvl in all_levels if lvl.price > current_price], key=lambda x: abs(x.distance_pct))[:4]
    return LevelSet(supports=supports, resistances=resistances, all_levels=all_levels)


def detect_candlestick_patterns(df: pd.DataFrame) -> List[str]:
    if len(df) < 2:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2]
    patterns = []

    o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
    po, pc = float(prev["Open"]), float(prev["Close"])
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if c > o and pc < po and c >= po and o <= pc:
        patterns.append("bullish engulfing candle")
    if c < o and pc > po and o >= pc and c <= po:
        patterns.append("bearish engulfing candle")
    if lower_wick > body * 2 and upper_wick < body and body / rng < 0.45:
        patterns.append("hammer-like rejection from lows")
    if upper_wick > body * 2 and lower_wick < body and body / rng < 0.45:
        patterns.append("shooting-star-like rejection from highs")
    if body / rng < 0.12:
        patterns.append("doji / indecision candle")

    return patterns


def detect_market_structure(df: pd.DataFrame, levels: LevelSet) -> List[str]:
    patterns = detect_candlestick_patterns(df)
    if len(df) < 60:
        return patterns

    last = df.iloc[-1]
    close = float(last["Close"])
    sma20 = float(last["SMA20"]) if np.isfinite(last["SMA20"]) else np.nan
    sma50 = float(last["SMA50"]) if np.isfinite(last["SMA50"]) else np.nan
    volume = float(last["Volume"])
    volume_ma = float(last["VolumeMA20"]) if np.isfinite(last["VolumeMA20"]) else 0

    recent_20_high = float(df["High"].tail(20).max())
    recent_20_low = float(df["Low"].tail(20).min())
    previous_20_high = float(df["High"].iloc[-40:-20].max()) if len(df) >= 40 else np.nan
    previous_20_low = float(df["Low"].iloc[-40:-20].min()) if len(df) >= 40 else np.nan

    if np.isfinite(sma20) and np.isfinite(sma50):
        if close > sma20 > sma50:
            patterns.append("short-term uptrend: close > SMA20 > SMA50")
        elif close < sma20 < sma50:
            patterns.append("short-term downtrend: close < SMA20 < SMA50")
        else:
            patterns.append("mixed / range-bound moving-average structure")

    if np.isfinite(previous_20_high) and recent_20_high > previous_20_high and close > sma20:
        patterns.append("recent higher-high attempt")
    if np.isfinite(previous_20_low) and recent_20_low < previous_20_low and close < sma20:
        patterns.append("recent lower-low pressure")

    if levels.resistances:
        nearest_resistance = levels.resistances[0].price
        if close > nearest_resistance and (volume_ma == 0 or volume > 1.2 * volume_ma):
            patterns.append("possible volume-confirmed resistance breakout")
    if levels.supports:
        nearest_support = levels.supports[0].price
        if close < nearest_support and (volume_ma == 0 or volume > 1.2 * volume_ma):
            patterns.append("possible volume-confirmed support breakdown")

    return patterns


# -----------------------------
# Signal and report generation
# -----------------------------

def calculate_signal(df: pd.DataFrame, levels: LevelSet) -> Signal:
    last = df.iloc[-1]
    close = float(last["Close"])
    rsi14 = float(last["RSI14"])
    macd_hist = float(last["MACDHist"])
    volume = float(last["Volume"])
    volume_ma = float(last["VolumeMA20"]) if np.isfinite(last["VolumeMA20"]) else 0
    atr14 = float(last["ATR14"]) if np.isfinite(last["ATR14"]) else close * 0.02

    score = 0
    reasons: List[str] = []

    def finite_value(value: Any) -> Optional[float]:
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except Exception:
            return None

    sma20 = finite_value(last.get("SMA20"))
    sma50 = finite_value(last.get("SMA50"))
    sma200 = finite_value(last.get("SMA200"))

    if sma20 is not None:
        if close > sma20:
            score += 1
            reasons.append("price is above SMA20")
        else:
            score -= 1
            reasons.append("price is below SMA20")

    if sma50 is not None:
        if close > sma50:
            score += 1
            reasons.append("price is above SMA50")
        else:
            score -= 1
            reasons.append("price is below SMA50")

    if sma20 is not None and sma50 is not None:
        if sma20 > sma50:
            score += 1
            reasons.append("SMA20 is above SMA50")
        else:
            score -= 1
            reasons.append("SMA20 is below SMA50")

    if sma200 is not None:
        if close > sma200:
            score += 1
            reasons.append("price is above SMA200")
        else:
            score -= 1
            reasons.append("price is below SMA200")

    if 50 <= rsi14 <= 70:
        score += 1
        reasons.append(f"RSI14 is constructive at {rsi14:.1f}")
    elif rsi14 > 75:
        score -= 1
        reasons.append(f"RSI14 is overbought at {rsi14:.1f}")
    elif rsi14 < 40:
        score -= 1
        reasons.append(f"RSI14 is weak at {rsi14:.1f}")
    else:
        reasons.append(f"RSI14 is neutral at {rsi14:.1f}")

    if macd_hist > 0:
        score += 1
        reasons.append("MACD histogram is positive")
    else:
        score -= 1
        reasons.append("MACD histogram is negative")

    if volume_ma > 0:
        if volume > 1.3 * volume_ma:
            reasons.append("latest volume is materially above its 20-bar average")
        elif volume < 0.7 * volume_ma:
            reasons.append("latest volume is below its 20-bar average")

    nearest_support = levels.supports[0] if levels.supports else None
    nearest_resistance = levels.resistances[0] if levels.resistances else None
    if nearest_support:
        distance_to_support = abs((close - nearest_support.price) / close)
        if distance_to_support <= max(0.01, atr14 / close):
            score += 1
            reasons.append(f"price is close to nearby support around {nearest_support.price:.2f}")
    if nearest_resistance:
        distance_to_resistance = abs((nearest_resistance.price - close) / close)
        if distance_to_resistance <= max(0.01, atr14 / close):
            score -= 1
            reasons.append(f"price is close to nearby resistance around {nearest_resistance.price:.2f}")

    if score >= 4:
        action = "BUY / ACCUMULATE ON CONFIRMATION"
        bias = "Bullish"
    elif score >= 2:
        action = "MILD BUY / HOLD"
        bias = "Constructive"
    elif score >= -1:
        action = "NEUTRAL / WAIT"
        bias = "Mixed"
    elif score >= -3:
        action = "SELL / REDUCE RISK"
        bias = "Bearish"
    else:
        action = "STRONG SELL / AVOID LONGS"
        bias = "Strongly bearish"

    if nearest_support and nearest_resistance:
        risk_note = (
            f"Nearest support: {nearest_support.price:.2f}; nearest resistance: {nearest_resistance.price:.2f}. "
            f"A practical invalidation level is often just below support, adjusted for ATR ({atr14:.2f})."
        )
    elif nearest_support:
        risk_note = f"Nearest support: {nearest_support.price:.2f}. Watch for a close below that area."
    elif nearest_resistance:
        risk_note = f"Nearest resistance: {nearest_resistance.price:.2f}. A breakout needs confirmation above that area."
    else:
        risk_note = "No reliable nearby support/resistance level was detected; reduce position size or wait for structure."

    return Signal(action=action, score=score, bias=bias, reasons=reasons[:8], risk_note=risk_note)


def format_levels(levels: List[Level]) -> str:
    if not levels:
        return "not enough reliable levels detected"
    return ", ".join([f"{lvl.price:.2f} ({abs(lvl.distance_pct):.1f}% away, {lvl.touches} touches)" for lvl in levels])


def build_local_report(ticker: str, timeframe_name: str, df: pd.DataFrame, levels: LevelSet, patterns: List[str], signal: Signal) -> str:
    last = df.iloc[-1]
    close = float(last["Close"])
    change = float(df["Close"].pct_change().iloc[-1] * 100) if len(df) > 1 else 0.0
    rsi14 = float(last["RSI14"])
    macd_hist = float(last["MACDHist"])
    atr14 = float(last["ATR14"]) if np.isfinite(last["ATR14"]) else close * 0.02
    volume = float(last["Volume"])
    volume_ma = float(last["VolumeMA20"]) if np.isfinite(last["VolumeMA20"]) else 0

    volume_text = "volume data is unavailable or not meaningful for this instrument"
    if volume_ma > 0:
        volume_text = f"latest volume is {volume / volume_ma:.2f}x the 20-bar average"

    pattern_text = "; ".join(patterns[:5]) if patterns else "no strong single-bar pattern was detected"

    report = f"""
### {timeframe_name} technical report for {ticker}

**Latest close:** {close:.2f} ({change:+.2f}% on the latest bar).  
**Trade bias:** **{signal.action}**. Internal score: **{signal.score}**, bias: **{signal.bias}**.

**Key levels.** Nearby support levels are {format_levels(levels.supports)}. Nearby resistance levels are {format_levels(levels.resistances)}. These levels are derived from clustered pivot highs/lows and recent range extremes, so they should be treated as zones rather than exact prices.

**Momentum and trend.** RSI14 is **{rsi14:.1f}**, MACD histogram is **{macd_hist:.4f}**, ATR14 is about **{atr14:.2f}**, and {volume_text}. The main evidence behind the signal is: {"; ".join(signal.reasons)}.

**Recent pattern read.** The current structure shows: {pattern_text}. If price holds above nearby support and then closes above resistance with stronger volume, the bullish scenario improves. If price loses support, especially with expanding volume, the bearish scenario becomes more important.

**Risk / trading note.** {signal.risk_note} This is a rule-based technical view, not financial advice. Confirm with your own risk controls, position sizing, and broader market context.
"""
    return report.strip()


def compact_payload_for_llm(ticker: str, timeframe_name: str, df: pd.DataFrame, levels: LevelSet, patterns: List[str], signal: Signal) -> Dict[str, Any]:
    last = df.iloc[-1]
    compact_bars = df.tail(40).copy()
    keep_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA20",
        "SMA50",
        "SMA200",
        "RSI14",
        "MACD",
        "MACDSignal",
        "MACDHist",
        "ATR14",
        "VolumeMA20",
    ]
    existing_cols = [c for c in keep_cols if c in compact_bars.columns]
    compact_bars = compact_bars[existing_cols].round(4)
    compact_bars.insert(0, "Datetime", compact_bars.index.astype(str))

    return {
        "ticker": ticker,
        "timeframe": timeframe_name,
        "latest": {
            "close": round(float(last["Close"]), 4),
            "rsi14": round(float(last["RSI14"]), 2),
            "macd_hist": round(float(last["MACDHist"]), 4),
            "atr14": round(float(last["ATR14"]), 4) if np.isfinite(last["ATR14"]) else None,
            "volume": round(float(last["Volume"]), 2),
        },
        "support_levels": [lvl.__dict__ for lvl in levels.supports],
        "resistance_levels": [lvl.__dict__ for lvl in levels.resistances],
        "detected_patterns": patterns,
        "rule_based_signal": signal.__dict__,
        "recent_bars": compact_bars.to_dict(orient="records"),
    }

def build_chatbot_prompt(
    ticker: str,
    timeframe_name: str,
    payload: Dict[str, Any],
    local_report: str,
) -> str:
    """Build a copy/paste prompt for ChatGPT, DeepSeek, Claude, Gemini, etc.

    This does not call any LLM API. The user manually copies this prompt into
    their preferred chatbot.
    """
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    return f"""
You are a cautious, disciplined technical analyst.

Analyze the following technical-analysis data for {ticker} on the {timeframe_name} timeframe.

Important rules:
- Use only the data provided below.
- Do not invent news, earnings, macro events, or price levels.
- Treat support and resistance as zones, not exact guaranteed prices.
- Do not claim certainty.
- Give bullish, bearish, and neutral scenarios.
- Explain trend, momentum, support/resistance, volume, and risk.
- The final output should be practical but not overconfident.
- This is not financial advice.

Technical-analysis payload:
{payload_json}

Rule-based technical analysis:
{local_report}

Please produce the final report with these sections:

1. Trend analysis
2. Momentum analysis
3. Support and resistance interpretation
4. Bullish scenario
5. Bearish scenario
6. Trading bias: bullish / neutral / bearish
7. Risk management notes
""".strip()


# -----------------------------
# Optional LLM integration
# -----------------------------

def call_llm_for_analysis(
    *,
    api_key: str,
    endpoint_url: str,
    model: str,
    payload: Dict[str, Any],
    timeout_seconds: int = 60,
) -> Optional[str]:
    """Call an OpenAI-compatible chat-completions endpoint.

    Replace this function if your LLM provider uses a different schema.

    Expected endpoint example:
        https://api.openai.com/v1/chat/completions

    Expected response shape:
        {"choices": [{"message": {"content": "..."}}]}
    """
    if not api_key or not endpoint_url or not model:
        return None

    system_prompt = (
        "You are a cautious technical-analysis assistant. "
        "Use only the supplied OHLCV and indicator data. "
        "Return a concise report with: trend, support/resistance, momentum, patterns, "
        "bullish scenario, bearish scenario, and a clear buy/hold/sell trading bias. "
        "Do not claim certainty. Include a risk-management note."
    )

    user_prompt = (
        "Please analyze this compact technical-analysis payload. "
        "Treat support/resistance as zones. Here is the JSON payload:\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(endpoint_url, headers=headers, json=body, timeout=timeout_seconds)
    response.raise_for_status()
    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, indent=2, ensure_ascii=False)


# -----------------------------
# Charting
# -----------------------------

def build_chart(ticker: str, timeframe_name: str, df: pd.DataFrame, levels: LevelSet, signal: Signal) -> go.Figure:
    visible = df.tail(180).copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=visible.index,
            open=visible["Open"],
            high=visible["High"],
            low=visible["Low"],
            close=visible["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    for col in ["SMA20", "SMA50", "SMA200"]:
        if col in visible.columns and visible[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=visible.index,
                    y=visible[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1.3),
                ),
                row=1,
                col=1,
            )

    if "BBUpper" in visible.columns and visible["BBUpper"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=visible.index,
                y=visible["BBUpper"],
                mode="lines",
                name="BB Upper",
                line=dict(width=0.8, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=visible.index,
                y=visible["BBLower"],
                mode="lines",
                name="BB Lower",
                line=dict(width=0.8, dash="dot"),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=visible.index,
            y=visible["Volume"],
            name="Volume",
            opacity=0.45,
        ),
        row=2,
        col=1,
    )

    if "VolumeMA20" in visible.columns and visible["VolumeMA20"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=visible.index,
                y=visible["VolumeMA20"],
                mode="lines",
                name="Volume MA20",
                line=dict(width=1.2),
            ),
            row=2,
            col=1,
        )

    # Support and resistance zones.
    for i, level in enumerate(levels.supports[:4], start=1):
        fig.add_hline(
            y=level.price,
            line_dash="dash",
            line_width=1,
            annotation_text=f"S{i}: {level.price:.2f}",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

    for i, level in enumerate(levels.resistances[:4], start=1):
        fig.add_hline(
            y=level.price,
            line_dash="dot",
            line_width=1,
            annotation_text=f"R{i}: {level.price:.2f}",
            annotation_position="top right",
            row=1,
            col=1,
        )

    latest = visible.iloc[-1]
    fig.add_annotation(
        x=visible.index[-1],
        y=float(latest["Close"]),
        text=signal.bias,
        showarrow=True,
        arrowhead=2,
        row=1,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker} — {timeframe_name} Technical Chart",
        xaxis_rangeslider_visible=False,
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# -----------------------------
# Streamlit app
# -----------------------------

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Input")
    user_ticker = st.sidebar.text_input(
        "Ticker or instrument",
        value="AAPL",
        help="Examples: AAPL, MSFT, 2899 HK, RY TSX, CBA AU, BTCUSD, EURUSD, SP500, GC=F",
    )

    st.sidebar.header("Analysis mode")

    analysis_mode = st.sidebar.radio(
        "Choose report mode",
        options=[
            "Rule-based technical analysis",
            "Generate prompt for chatbot",
            "BYO API directly",
        ],
        index=0,
        help=(
            "Prompt mode does not call any API. It only generates a prompt that users "
            "can copy into ChatGPT, DeepSeek, Claude, Gemini, etc."
        ),
    )

    use_llm = analysis_mode == "BYO API directly"
    use_chatbot_prompt = analysis_mode == "Generate prompt for chatbot"

    endpoint_url = ""
    model = ""
    api_key = ""

    if use_llm:
        st.sidebar.markdown("### LLM API settings")

        endpoint_url = st.sidebar.text_input(
            "LLM endpoint URL",
            value="",
            placeholder="Example: https://api.deepseek.com/chat/completions",
            help="Enter the full chat-completions endpoint URL from your LLM provider.",
        )

        model = st.sidebar.text_input(
            "Model name",
            value="",
            placeholder="Example: deepseek-chat, deepseek-reasoner, gpt-4o-mini",
            help="Enter the exact model name provided by your LLM provider.",
        )

        api_key = st.sidebar.text_input(
            "API key",
            value="",
            type="password",
            help="Your API key is used only to call the selected LLM endpoint.",
        )

        st.sidebar.caption(
            "API mode sends the technical-analysis payload to your selected LLM endpoint. "
            "Do not enter a production API key unless you trust this app/server."
        )

    if use_chatbot_prompt:
        st.sidebar.caption(
            "Prompt mode has no API cost and no API-key leakage risk. "
            "The app generates a prompt; users copy/paste it into their own chatbot."
        )

    st.sidebar.header("Options")
    show_local_report_with_llm = st.sidebar.checkbox(
        "Also show rule-based analysis",
        value=True,
    )
    run_button = st.sidebar.button("Generate reports", type="primary")

    return {
        "user_ticker": user_ticker,
        "analysis_mode": analysis_mode,
        "use_llm": use_llm,
        "use_chatbot_prompt": use_chatbot_prompt,
        "endpoint_url": endpoint_url,
        "model": model,
        "api_key": api_key,
        "show_local_report_with_llm": show_local_report_with_llm,
        "run_button": run_button,
    }


def render_report_for_timeframe(
    *,
    ticker: str,
    timeframe_name: str,
    settings: Dict[str, Any],
    use_llm: bool,
    use_chatbot_prompt: bool,
    api_key: str,
    endpoint_url: str,
    model: str,
    show_local_report_with_llm: bool,
) -> None:
    with st.spinner(f"Fetching and analyzing {timeframe_name} data..."):
        raw = fetch_market_data(ticker, settings["period"], settings["interval"])
        df = add_indicators(raw)
        if len(df) < 60:
            st.warning(f"Only {len(df)} bars were returned. Some indicators and levels may be unreliable.")

        levels = calculate_levels(
            df,
            order=settings["pivot_order"],
            lookback_bars=settings["lookback_bars_for_levels"],
        )
        patterns = detect_market_structure(df, levels)
        signal = calculate_signal(df, levels)
        local_report = build_local_report(ticker, timeframe_name, df, levels, patterns, signal)
        chart = build_chart(ticker, timeframe_name, df, levels, signal)

    st.plotly_chart(chart, use_container_width=True)

    if use_chatbot_prompt:
        payload = compact_payload_for_llm(ticker, timeframe_name, df, levels, patterns, signal)
        chatbot_prompt = build_chatbot_prompt(
            ticker=ticker,
            timeframe_name=timeframe_name,
            payload=payload,
            local_report=local_report,
        )

        st.markdown("### Copy/paste prompt for ChatGPT / DeepSeek / Claude / Gemini")
        st.caption(
            "No LLM API is called here. Copy this prompt into your own chatbot to get an AI-written report."
        )

        st.text_area(
            "Prompt",
            value=chatbot_prompt,
            height=420,
            key=f"chatbot_prompt_{ticker}_{timeframe_name}",
        )

        st.download_button(
            label="Download prompt as TXT",
            data=chatbot_prompt,
            file_name=f"{ticker}_{timeframe_name}_ta_prompt.txt".replace("/", "_"),
            mime="text/plain",
            key=f"download_prompt_{ticker}_{timeframe_name}",
        )

        with st.expander("Rule-based technical analysis"):
            st.markdown(local_report)

        return

    if use_llm:
        if not api_key or not endpoint_url or not model:
            st.warning("LLM is enabled, but endpoint URL, model, or API key is missing. Showing local report instead.")
            st.markdown(local_report)
            return

        payload = compact_payload_for_llm(ticker, timeframe_name, df, levels, patterns, signal)
        try:
            with st.spinner(f"Calling LLM for {timeframe_name} report..."):
                llm_report = call_llm_for_analysis(
                    api_key=api_key,
                    endpoint_url=endpoint_url,
                    model=model,
                    payload=payload,
                )
            if llm_report:
                st.markdown("### AI-generated analysis")
                st.markdown(llm_report)
                if show_local_report_with_llm:
                    with st.expander("Rule-based technical analysis"):
                        st.markdown(local_report)
            else:
                st.warning("LLM returned no report. Showing rule-based technical analysis instead.")
                st.markdown(local_report)
        except Exception as exc:
            st.error(f"LLM call failed: {exc}")
            st.markdown(local_report)
    else:
        st.markdown(local_report)


def main() -> None:
    st.set_page_config(page_title="Technical Analysis Reports", layout="wide")
    st.title("Technical Analysis on your ticker")
    st.caption("Daily, weekly and 15-minute reports. For research and education only — not financial advice.")

    options = render_sidebar()

    st.markdown(
        """
This app pulls OHLCV data from Yahoo Finance through `yfinance`, calculates common technical indicators, finds support/resistance zones, and generates daily/weekly/15-minute technical analysis reports.

To have LLM do more analysis, you can generate prompt and copy/paste to your chatbot, ore BYO API.

**Ticker examples:** `AAPL`, `MSFT`, `2899 HK`, `RY TSX`, `CBA AU`, `BTCUSD`, `EURUSD`, `SP500`, `GC=F`.
"""
    )

    if not options["run_button"]:
        st.info("Enter a ticker in the sidebar and click **Generate reports**.")
        return

    if not options["user_ticker"].strip():
        st.error("Please enter a ticker.")
        return

    with st.spinner("Resolving ticker symbol..."):
        ticker = resolve_ticker(options["user_ticker"])

    st.success(f"Using yfinance ticker: `{ticker}`")

    ordered_timeframes = ["Daily", "Weekly", "15-Minute"]

    tabs = st.tabs(ordered_timeframes)
    for tab, timeframe_name in zip(tabs, ordered_timeframes):
        with tab:
            st.subheader(f"{timeframe_name} report")
            try:
                render_report_for_timeframe(
                    ticker=ticker,
                    timeframe_name=timeframe_name,
                    settings=TIMEFRAMES[timeframe_name],
                    use_llm=options["use_llm"],
                    use_chatbot_prompt=options["use_chatbot_prompt"],
                    api_key=options["api_key"],
                    endpoint_url=options["endpoint_url"],
                    model=options["model"],
                    show_local_report_with_llm=options["show_local_report_with_llm"],
                )
            except Exception as exc:
                st.error(f"Could not generate {timeframe_name} report: {exc}")


if __name__ == "__main__":
    main()
