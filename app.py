import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from itertools import product
import json
import requests
import os
import socket
import sys
import time
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent
CACHE_FILE    = BASE_DIR / "backup_data.csv"
CACHE_METADATA_FILE = BASE_DIR / "backup_data.meta.json"
REAL_YIELD_CACHE_FILE = BASE_DIR / "real_yield_cache.csv"
REAL_YIELD_SEED_FILE = BASE_DIR / "data" / "real_yield_seed.csv"
CACHE_MAX_AGE = 23   # hours
REAL_YIELD_WARNING_AGE_DAYS = 4
TRADING_DAYS_PER_YEAR = 365
WALK_FORWARD_MIN_TRAIN = 365
WALK_FORWARD_WINDOW = 730
ALPHA_MIN_TRAIN = 180
REOPTIMIZE_EVERY_DAYS = 7
REQUIRED_COLS = ["BTC", "NASDAQ", "DXY", "GOLD", "OIL", "REAL_YIELD"]
YFINANCE_TICKERS = [("QQQ", "NASDAQ"), ("DX-Y.NYB", "DXY"), ("GC=F", "GOLD"), ("CL=F", "OIL")]
NASDAQ_FALLBACK_TICKERS = [
    ("QQQ", "NASDAQ"),
    ("UUP", "DXY"),
    ("GLD", "GOLD"),
    ("USO", "OIL"),
]
COMMON_LOCAL_PROXY_PORTS = (7890, 7897, 10809, 1080)

# ── PROXY CONFIG (required for mainland China) ───────────────────────────────
# Proxy is optional. Explicit app settings take priority, followed by standard
# proxy environment variables and a quick local-port check for desktop use.
def get_setting(name):
    value = os.environ.get(name)
    if value not in (None, ""):
        return value
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value


def normalize_proxy_url(value):
    if value in (None, "", "0", "none", "None", "false", "False"):
        return None
    value = str(value).strip()
    if "://" not in value:
        value = f"http://{value}"
    return value


def local_port_is_open(port, timeout=0.08):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def get_proxy_config():
    explicit_url = normalize_proxy_url(get_setting("BTC_PROXY_URL"))
    if explicit_url:
        return explicit_url, "BTC_PROXY_URL"

    explicit_port = get_setting("BTC_PROXY_PORT") or get_setting("PROXY_PORT")
    if explicit_port not in (None, ""):
        try:
            port = int(explicit_port)
            return f"http://127.0.0.1:{port}", "BTC_PROXY_PORT"
        except (TypeError, ValueError):
            pass

    for env_name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"):
        env_proxy = normalize_proxy_url(os.environ.get(env_name))
        if env_proxy:
            return env_proxy, env_name

    auto_detect = get_setting("BTC_PROXY_AUTO_DETECT")
    if auto_detect not in ("0", "false", "False", "no", "No"):
        for port in COMMON_LOCAL_PROXY_PORTS:
            if local_port_is_open(port):
                return f"http://127.0.0.1:{port}", f"auto-detected local port {port}"

    return None, "direct connection"


PROXY_URL, PROXY_SOURCE = get_proxy_config()

SESSION = requests.Session()
SESSION.trust_env = False
if PROXY_URL:
    SESSION.proxies.update({"http": PROXY_URL, "https": PROXY_URL})

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📊 BTC Macro Driver Dashboard")

manual_refresh = st.sidebar.button("🔄 Refresh Data")
transaction_cost_bps = st.sidebar.number_input(
    "Backtest cost (bps/trade)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
)
show_timing = st.sidebar.checkbox("Show timing", value=False)


@contextmanager
def timed(label, timings=None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if timings is not None:
            timings[label] = elapsed
        elif show_timing:
            st.sidebar.write(f"{label}: {elapsed:.2f}s")

if manual_refresh:
    st.cache_data.clear()

# ─────────────────────────────────────────────
# SMART CACHE
# ─────────────────────────────────────────────

def cache_is_fresh():
    if not os.path.exists(CACHE_FILE):
        return False
    age_hours = (time.time() - os.path.getmtime(CACHE_FILE)) / 3600
    if age_hours >= CACHE_MAX_AGE:
        return False
    try:
        cached = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        if cached.empty or any(c not in cached.columns for c in REQUIRED_COLS):
            return False
        latest_observation = pd.Timestamp(cached.index[-1]).normalize()
        observation_age_days = (pd.Timestamp.now().normalize() - latest_observation).days
        return observation_age_days <= 2
    except Exception:
        return False

def load_cache():
    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    status = {col: "cache" for col in df.columns}
    timestamps = {col: df.index[-1] for col in df.columns}
    if CACHE_METADATA_FILE.exists():
        try:
            metadata = json.loads(CACHE_METADATA_FILE.read_text(encoding="utf-8"))
            source_status = metadata.get("status", {})
            for col in df.columns:
                if col in source_status:
                    status[col] = f"cache; last source: {source_status[col]}"
            if "_macro_info" in source_status:
                status["_macro_info"] = source_status["_macro_info"]
            for col, value in metadata.get("timestamps", {}).items():
                timestamps[col] = pd.Timestamp(value)
        except Exception:
            pass
    if REAL_YIELD_CACHE_FILE.exists():
        try:
            real_yield_cache = pd.read_csv(
                REAL_YIELD_CACHE_FILE,
                index_col=0,
                parse_dates=True,
            ).dropna()
            if not real_yield_cache.empty:
                timestamps["REAL_YIELD"] = real_yield_cache.index[-1]
        except Exception:
            pass
    return df, status, timestamps


def save_cache(data, status, timestamps):
    data.to_csv(CACHE_FILE)
    metadata = {
        "status": {
            key: value
            for key, value in status.items()
            if key in REQUIRED_COLS or key == "_macro_info"
        },
        "timestamps": {
            key: str(pd.Timestamp(value))
            for key, value in timestamps.items()
            if value is not None
        },
    }
    CACHE_METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_cache_if_complete():
    if not os.path.exists(CACHE_FILE):
        return None, None, None
    data, status, timestamps = load_cache()
    if data.empty or any(c not in data.columns for c in REQUIRED_COLS):
        return None, None, None
    return data, status, timestamps

# ─────────────────────────────────────────────
# BTC LOADERS — each returns (series_or_None, error_str_or_None)
# Huobi is primary; all others are fallbacks.
# ─────────────────────────────────────────────

def load_btc_huobi():
    """Huobi HTX — no auth, China-accessible, up to 2000 daily candles."""
    try:
        url = "https://api.huobi.pro/market/history/kline"
        r = SESSION.get(url, params={"symbol": "btcusdt", "period": "1day", "size": 2000}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok" or "data" not in data:
            return None, f"bad response body: {str(data)[:150]}"
        df = pd.DataFrame(data["data"])
        df["time"] = pd.to_datetime(df["id"], unit="s").dt.normalize()
        df = df.set_index("time").sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df["close"].astype(float), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def load_btc_okx():
    """OKX — no auth, China-accessible, last 100 daily candles."""
    try:
        url = "https://www.okx.com/api/v5/market/history-candles"
        r = SESSION.get(url, params={"instId": "BTC-USDT", "bar": "1D", "limit": "100"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "0" or not data.get("data"):
            return None, f"bad response body: {str(data)[:150]}"
        df = pd.DataFrame(data["data"])
        df["time"] = pd.to_datetime(df[0].astype(float), unit="ms").dt.normalize()
        df = df.set_index("time").sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df[4].astype(float).rename("close"), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def load_btc_binance():
    """Binance — global fallback, may be blocked in China."""
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000"
        r = SESSION.get(url, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json()).iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["time"] = pd.to_datetime(df["time"], unit="ms").dt.normalize()
        df = df.set_index("time")
        return df["close"].astype(float), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def load_btc_coingecko():
    """CoinGecko — last resort, heavily rate-limited on free tier."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        r = SESSION.get(url, params={"vs_currency": "usd", "days": "max"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "prices" not in data:
            return None, f"no prices key: {str(data)[:150]}"
        df = pd.DataFrame(data["prices"], columns=["time", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms").dt.normalize()
        df = df.set_index("time").sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep="last")]
        return df["close"].astype(float), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def load_btc():
    """Try all sources; return (series, source_name, errors_dict)."""
    errors = {}

    series, err = load_btc_huobi()
    if series is not None:
        return series, "huobi", errors
    errors["Huobi"] = err

    series, err = load_btc_okx()
    if series is not None:
        return series, "okx", errors
    errors["OKX"] = err

    series, err = load_btc_binance()
    if series is not None:
        return series, "binance", errors
    errors["Binance"] = err

    series, err = load_btc_coingecko()
    if series is not None:
        return series, "coingecko", errors
    errors["CoinGecko"] = err

    return None, None, errors

# ─────────────────────────────────────────────
# MACRO DATA LOADER (yfinance via proxy)
# Replaced Stooq — yfinance routes through SESSION proxy reliably.
# Ticker map:
#   QQQ      → NASDAQ-100 proxy
#   DX-Y.NYB → US Dollar Index (DXY)
#   GC=F     → Gold futures
#   CL=F     → WTI Crude Oil futures
# ─────────────────────────────────────────────

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def normalize_daily_index(index):
    """Return timezone-naive daily timestamps so all sources align by date."""
    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def restore_env_var(name, old_value):
    if old_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old_value


def extract_yfinance_close(raw, ticker):
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            frame = raw[ticker]
        elif ticker in raw.columns.get_level_values(-1):
            frame = raw.xs(ticker, axis=1, level=-1)
        else:
            return None
    else:
        frame = raw

    if "Close" not in frame.columns:
        return None

    close = frame["Close"].dropna().squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if close.empty:
        return None

    close.index = normalize_daily_index(close.index)
    close = close[~close.index.duplicated(keep="last")]
    return close.astype(float)


def load_macro_yfinance_batch(start_date):
    """
    Download all macro series in one threaded yfinance request.
    yfinance uses curl_cffi internally, so proxy settings are applied through
    temporary HTTP_PROXY / HTTPS_PROXY env vars and then restored.
    """
    statuses = {}
    series_by_name = {}
    timestamps = {}

    if not YFINANCE_AVAILABLE:
        msg = "yfinance not installed — run: pip install yfinance"
        return series_by_name, {name: msg for _, name in YFINANCE_TICKERS}, timestamps

    try:
        old_http = os.environ.get("HTTP_PROXY")
        old_https = os.environ.get("HTTPS_PROXY")
        if PROXY_URL:
            os.environ["HTTP_PROXY"] = PROXY_URL
            os.environ["HTTPS_PROXY"] = PROXY_URL

        try:
            try:
                yf.config.network.proxy = PROXY_URL
                yf.config.network.retries = 2
            except (AttributeError, TypeError):
                pass

            raw = yf.download(
                [ticker for ticker, _ in YFINANCE_TICKERS],
                start=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                threads=True,
                progress=False,
                timeout=10,
                group_by="ticker",
            )
        finally:
            restore_env_var("HTTP_PROXY", old_http)
            restore_env_var("HTTPS_PROXY", old_https)

        if raw is None or raw.empty:
            return series_by_name, {name: "empty response" for _, name in YFINANCE_TICKERS}, timestamps

        for ticker, name in YFINANCE_TICKERS:
            close = extract_yfinance_close(raw, ticker)
            if close is None or close.empty:
                statuses[name] = "empty response"
                continue
            series_by_name[name] = close.rename(name)
            statuses[name] = "live"
            timestamps[name] = close.index[-1]

        return series_by_name, statuses, timestamps
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        return series_by_name, {name: msg for _, name in YFINANCE_TICKERS}, timestamps

# ─────────────────────────────────────────────
# MAIN DATA LOADER
# ─────────────────────────────────────────────

def load_macro_nasdaq_etfs(start_date):
    """
    Load a consistent ETF proxy set from Nasdaq when Yahoo is unavailable.

    UUP, GLD and USO are not identical to DXY, gold futures and WTI futures.
    The fallback replaces the full feature history rather than splicing proxy
    returns onto the original instruments.
    """
    series_by_name = {}
    statuses = {}
    timestamps = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/market-activity/",
    }
    start_text = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_text = pd.Timestamp.now().strftime("%Y-%m-%d")

    for ticker, name in NASDAQ_FALLBACK_TICKERS:
        try:
            response = SESSION.get(
                f"https://api.nasdaq.com/api/quote/{ticker}/historical",
                params={
                    "assetclass": "etf",
                    "fromdate": start_text,
                    "todate": end_text,
                    "limit": "5000",
                },
                headers=headers,
                timeout=25,
            )
            response.raise_for_status()
            payload = response.json()
            rows = (
                payload.get("data", {})
                .get("tradesTable", {})
                .get("rows", [])
            )
            if not rows:
                raise ValueError("Nasdaq returned no historical rows")

            frame = pd.DataFrame(rows)
            dates = pd.to_datetime(frame["date"], errors="coerce")
            closes = pd.to_numeric(
                frame["close"].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            )
            series = pd.Series(closes.to_numpy(), index=dates, name=name).dropna()
            series.index = normalize_daily_index(series.index)
            series = series[~series.index.duplicated(keep="last")].sort_index()
            if series.empty:
                raise ValueError("Nasdaq close history was not numeric")

            series_by_name[name] = series
            proxy_label = ticker if ticker == "QQQ" else f"{ticker} proxy"
            statuses[name] = f"live (Nasdaq {proxy_label})"
            timestamps[name] = series.index[-1]
        except Exception as exc:
            statuses[name] = f"failed Nasdaq {ticker} ({type(exc).__name__}: {exc})"

    return series_by_name, statuses, timestamps


def clean_real_yield(raw):
    """Normalize FRED/cache data into a dated REAL_YIELD series."""
    if raw is None or raw.empty:
        return None

    frame = raw.copy()
    if "REAL_YIELD" in frame.columns:
        value_col = "REAL_YIELD"
    elif "DFII10" in frame.columns:
        value_col = "DFII10"
    elif len(frame.columns) == 1:
        value_col = frame.columns[0]
    else:
        return None

    series = pd.to_numeric(frame[value_col], errors="coerce").dropna()
    if series.empty:
        return None

    series.index = normalize_daily_index(series.index)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.astype(float).rename("REAL_YIELD")


def read_real_yield_fallback():
    """Use an exact last-known FRED observation; never estimate the yield."""
    candidates = [
        (REAL_YIELD_CACHE_FILE, "real-yield cache"),
        (REAL_YIELD_SEED_FILE, "bundled official seed"),
        (CACHE_FILE, "complete-data cache"),
    ]
    errors = []

    for path, label in candidates:
        if not path.exists():
            continue
        try:
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
            series = clean_real_yield(cached)
            if series is not None:
                if path == CACHE_FILE:
                    series = series[series.index.dayofweek < 5]
                return series, label, errors
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    return None, None, errors


def load_treasury_real_yield(start_year):
    """Load the official U.S. Treasury 10-year par real yield by calendar year."""
    pieces = []
    current_year = pd.Timestamp.now().year
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        "interest-rates/TextView"
    )

    for year in range(max(2003, int(start_year)), current_year + 1):
        response = SESSION.get(
            url,
            params={
                "type": "daily_treasury_real_yield_curve",
                "field_tdr_date_value": str(year),
            },
            headers=headers,
            timeout=40,
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text), match="10 YR")
        if not tables:
            continue
        frame = tables[0]
        if "Date" not in frame.columns or "10 YR" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce")
        values = pd.to_numeric(frame["10 YR"], errors="coerce")
        piece = pd.Series(values.to_numpy(), index=dates, name="REAL_YIELD").dropna()
        if not piece.empty:
            pieces.append(piece)

    if not pieces:
        return None

    series = pd.concat(pieces).sort_index()
    series.index = normalize_daily_index(series.index)
    return series[~series.index.duplicated(keep="last")].astype(float)


def load_real_yield():
    """
    Load the daily FRED DFII10 series through the proxy-aware session.

    FRED publishes on business days. Weekend scoring therefore uses the exact
    last published observation, while its source date remains visible.
    """
    errors = []
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"

    try:
        response = SESSION.get(url, timeout=(5, 12))
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
        series = clean_real_yield(raw)
        if series is None:
            raise ValueError("FRED returned no numeric DFII10 observations")
        series.to_frame().to_csv(REAL_YIELD_CACHE_FILE)
        return series, "live", series.index[-1], errors
    except Exception as exc:
        errors.append(f"FRED: {type(exc).__name__}: {exc}")

    fallback, label, fallback_errors = read_real_yield_fallback()
    errors.extend(fallback_errors)

    try:
        if fallback is None or fallback.empty:
            treasury_start_year = 2020
        else:
            treasury_start_year = fallback.index[-1].year
        treasury = load_treasury_real_yield(treasury_start_year)
        if treasury is not None:
            if fallback is not None:
                treasury = pd.concat([fallback, treasury]).sort_index()
                treasury = treasury[~treasury.index.duplicated(keep="last")]
            treasury.to_frame().to_csv(REAL_YIELD_CACHE_FILE)
            return (
                treasury,
                "live (U.S. Treasury fallback)",
                treasury.index[-1],
                errors,
            )
    except Exception as exc:
        errors.append(f"U.S. Treasury: {type(exc).__name__}: {exc}")

    if fallback is not None:
        return fallback, f"stale {label}", fallback.index[-1], errors

    return None, "failed", None, errors


@st.cache_data(ttl=900)
def load_data(force_refresh=False, network_route=None):
    del network_route

    # Use cache if fresh — avoids hitting APIs on every reload
    if not force_refresh and cache_is_fresh():
        data, status, timestamps = load_cache()
        status["_info"] = f"loaded from cache (< {CACHE_MAX_AGE}h old). Click Refresh to force update."
        return data, status, timestamps

    data = pd.DataFrame()
    status = {}
    timestamps = {}

    # ── BTC ──────────────────────────────────
    btc_series, btc_source, btc_errors = load_btc()

    if btc_series is None:
        # Show exactly why every source failed
        error_lines = "\n".join([f"  • {src}: {msg}" for src, msg in btc_errors.items()])
        if os.path.exists(CACHE_FILE):
            data, _, cached_timestamps = load_cache()
            timestamps.update(cached_timestamps)
            status["BTC"] = "stale cache (all live sources failed)"
            status["_btc_errors"] = error_lines
        else:
            st.error("❌ All BTC sources failed and no cache exists.")
            st.expander("🔍 Diagnostic — exact errors per source").write(error_lines)
            st.stop()
            raise SystemExit
    else:
        data["BTC"] = btc_series
        status["BTC"] = f"live ({btc_source})"
        timestamps["BTC"] = data.index[-1]
        # Still show errors for sources that were tried before success
        if btc_errors:
            status["_btc_errors"] = "\n".join([f"  • {s}: {m}" for s, m in btc_errors.items()])

    # ── MACRO (batched yfinance via proxy) ───
    macro_start = data["BTC"].index.min()
    macro_series, macro_status, macro_timestamps = load_macro_yfinance_batch(macro_start)
    expected_macro_names = {name for _, name in YFINANCE_TICKERS}
    if set(macro_series) != expected_macro_names:
        yahoo_diagnostics = "\n".join(
            f"{name}: {message}" for name, message in macro_status.items()
        )
        nasdaq_series, nasdaq_status, nasdaq_timestamps = load_macro_nasdaq_etfs(macro_start)
        if set(nasdaq_series) == expected_macro_names:
            macro_series = nasdaq_series
            macro_status = nasdaq_status
            macro_timestamps = nasdaq_timestamps
            status["_macro_info"] = (
                "Yahoo was unavailable; using a consistent Nasdaq ETF proxy set "
                "(QQQ, UUP, GLD, USO) for the full model history."
            )
            status["_macro_errors"] = yahoo_diagnostics

    status.update(macro_status)
    timestamps.update(macro_timestamps)
    for name, series in macro_series.items():
        data[name] = series

    # ── FRED REAL YIELD ──────────────────────
    real_yield, real_yield_status, real_yield_timestamp, real_yield_errors = load_real_yield()
    status["REAL_YIELD"] = real_yield_status
    if real_yield_errors:
        status["_real_yield_errors"] = "\n".join(real_yield_errors)
    if real_yield_timestamp is not None:
        timestamps["REAL_YIELD"] = real_yield_timestamp
    if real_yield is not None:
        if "REAL_YIELD" in data.columns:
            data = data.drop(columns=["REAL_YIELD"])
        data = data.join(real_yield, how="left")

    missing_live = [c for c in REQUIRED_COLS if c not in data.columns]
    if missing_live:
        cached_data, _, cached_timestamps = load_cache_if_complete()
        if cached_data is not None:
            status["_info"] = f"live refresh incomplete ({missing_live}); showing last complete cache."
            for column in REQUIRED_COLS:
                status[column] = "stale complete-data cache"
            return cached_data, status, cached_timestamps

    # ── CLEAN + SAVE ─────────────────────────
    data = data.sort_index()

    # 🔥 Align all series to BTC range
    start = data["BTC"].index.min()
    end = data["BTC"].index.max()

    data = data.loc[start:end]

    # Forward-fill macro series instead of requiring a perfect overlap
    macro_cols = [c for c in REQUIRED_COLS if c != "BTC"]
    for c in macro_cols:
        if c in data.columns:
            data[c] = data[c].ffill()

    # Fill after alignment, then keep only dates where required data exists.
    data = data.ffill()
    data = data.dropna(subset=["BTC"])
    data = data.dropna(subset=[c for c in REQUIRED_COLS if c in data.columns])

    if data.empty:
        cached_data, _, cached_timestamps = load_cache_if_complete()
        if cached_data is not None:
            status["_info"] = "live refresh returned no overlapping rows; showing last complete cache."
            for column in REQUIRED_COLS:
                status[column] = "stale complete-data cache"
            return cached_data, status, cached_timestamps

    # Optional: show coverage so you can see which series is too sparse
    st.sidebar.write("### 📊 Coverage")
    for c in REQUIRED_COLS:
        if c in data.columns:
            st.sidebar.write(f"{c}: {data[c].notna().sum()} rows")

    if not data.empty:
        save_cache(data, status, timestamps)

    return data, status, timestamps

if show_timing:
    st.sidebar.write("### Timing")

with timed("load_data"):
    data, status, timestamps = load_data(
        force_refresh=manual_refresh,
        network_route=PROXY_URL or "direct",
    )
st.sidebar.write("### 🧪 Index Debug")

for col in data.columns:
    st.sidebar.write(
        f"{col}: {data[col].index.min()} → {data[col].index.max()} | rows={len(data[col].dropna())}"
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.write("### Network")
if PROXY_URL:
    st.sidebar.success(f"Proxy active: {PROXY_URL}")
    st.sidebar.caption(PROXY_SOURCE)
else:
    st.sidebar.warning("Direct connection. Set BTC_PROXY_PORT if live sources time out.")

st.sidebar.write("### 📡 Data Status")
for k, v in status.items():
    if k == "_info":
        st.sidebar.info(f"ℹ️ {v}")
    elif k == "_btc_errors":
        with st.sidebar.expander("🔍 BTC source errors"):
            st.text(v)
    elif k == "_real_yield_errors":
        with st.sidebar.expander("Real-yield source errors"):
            st.text(v)
    elif k == "_macro_info":
        st.sidebar.info(v)
    elif k == "_macro_errors":
        with st.sidebar.expander("Yahoo source errors"):
            st.text(v)
    else:
        status_text = str(v).lower()
        if status_text.startswith("live"):
            icon = "OK"
        elif "stale" in status_text:
            icon = "STALE"
        elif "cache" in status_text:
            icon = "CACHED"
        else:
            icon = "FAIL"
        st.sidebar.write(f"{icon} {k}: {v}")

st.sidebar.write("### 🕒 Freshness")
for k, t in timestamps.items():
    st.sidebar.write(f"{k}: {t}")

real_yield_as_of = timestamps.get("REAL_YIELD")
if real_yield_as_of is not None:
    real_yield_age_days = max(
        0,
        (pd.Timestamp.now().normalize() - pd.Timestamp(real_yield_as_of).normalize()).days,
    )
    if real_yield_age_days > REAL_YIELD_WARNING_AGE_DAYS:
        st.sidebar.warning(
            f"Real yield is {real_yield_age_days} calendar days old. "
            "Model inputs are stale; treat the signal as historical."
        )

btc_observations = data["BTC"].dropna() if "BTC" in data.columns else pd.Series(dtype=float)
data_as_of = btc_observations.index[-1] if not btc_observations.empty else None
data_age_days = None
if data_as_of is not None:
    data_age_days = max(
        0,
        (pd.Timestamp.now().normalize() - pd.Timestamp(data_as_of).normalize()).days,
    )
    st.sidebar.write(f"Displayed data as of: {pd.Timestamp(data_as_of):%Y-%m-%d}")

fallback_mode = any(
    "stale" in str(value).lower()
    for key, value in status.items()
    if not key.startswith("_")
)
if data_age_days is not None and data_age_days > 2:
    st.error(
        f"Historical fallback mode: displayed market data ends "
        f"{pd.Timestamp(data_as_of):%Y-%m-%d} ({data_age_days} days old). "
        "Charts and backtests remain available, but the signal is not current."
    )
elif fallback_mode:
    st.warning(
        "One or more live feeds are unavailable. Cached observations are being "
        "used and are labeled with their source dates."
    )
st.sidebar.write(f"🕒 Run Time: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.write(f"📊 Observations: {len(data)}")

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
required = REQUIRED_COLS
missing = [c for c in required if c not in data.columns]
if missing:
    st.error(f"Missing columns: {missing}. Check sidebar for failed sources.")
    st.stop()
    raise SystemExit


def safe_rolling_z(series, window=90, min_periods=20):
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-3, 3)


def classify_regime(score):
    if score >= 58:
        return "Risk-on"
    if score <= 42:
        return "Risk-off"
    return "Mixed"


def classify_signal(score):
    if score >= 65:
        return "BUY"
    if score <= 35:
        return "SELL"
    return "NEUTRAL"


def annualized_sharpe(ret):
    ret = ret.dropna()
    if len(ret) < 2 or ret.std() == 0:
        return np.nan
    return np.sqrt(TRADING_DAYS_PER_YEAR) * ret.mean() / ret.std()


def annualized_sharpe_array(ret):
    ret = np.asarray(ret, dtype=float)
    ret = ret[np.isfinite(ret)]
    if ret.size < 2:
        return np.nan
    std = ret.std(ddof=1)
    if std == 0:
        return np.nan
    return np.sqrt(TRADING_DAYS_PER_YEAR) * ret.mean() / std


def strategy_returns_from_score_array(score, target, buy_threshold, sell_threshold, cost_rate):
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=float)
    position = np.select([score >= buy_threshold, score <= sell_threshold], [1.0, -1.0], default=0.0)
    turnover = np.abs(np.diff(position, prepend=0.0))
    return position * target - turnover * cost_rate


def max_drawdown(equity):
    if equity.empty:
        return np.nan
    return (equity / equity.cummax() - 1).min()


def format_metric(value, fmt="{:.2f}"):
    if pd.isna(value):
        return "n/a"
    return fmt.format(value)


def normalize_abs_weights(values):
    weights = pd.Series(values).astype(float).fillna(0)
    denom = weights.abs().sum()
    if denom == 0:
        return weights * 0
    return weights / denom


def walk_forward_linear_model(features, target, min_train, train_window):
    frame = features.join(target.rename("target")).dropna()
    x = frame[features.columns]
    y_target = frame["target"]
    x_values = x.to_numpy(dtype=float)
    y_values = y_target.to_numpy(dtype=float)
    prediction_values = np.full(len(features.index), np.nan)
    coef_values = np.full((len(features.index), len(features.columns)), np.nan)
    output_locs = features.index.get_indexer(x.index)

    for i in range(min_train, len(frame)):
        start = max(0, i - train_window) if train_window else 0
        x_train = x_values[start:i]
        y_train = y_values[start:i]
        design = np.column_stack([np.ones(len(x_train)), x_train])
        beta, *_ = np.linalg.lstsq(design, y_train, rcond=None)
        out_pos = output_locs[i]
        if out_pos >= 0:
            prediction_values[out_pos] = np.r_[1.0, x_values[i]] @ beta
            coef_values[out_pos] = beta[1:]

    predictions = pd.Series(prediction_values, index=features.index, name="Walk-Forward Prediction")
    coefs = pd.DataFrame(coef_values, index=features.index, columns=features.columns)
    return predictions, coefs


def candidate_weight_grid(n_features):
    candidates = []
    for combo in product([-1.0, -0.5, 0.0, 0.5, 1.0], repeat=n_features):
        weights = np.array(combo, dtype=float)
        denom = np.abs(weights).sum()
        if denom == 0:
            continue
        candidates.append(weights / denom)
    return candidates


def strategy_returns_from_score(score, target, buy_threshold, sell_threshold, cost_rate):
    position = pd.Series(
        np.select([score >= buy_threshold, score <= sell_threshold], [1, -1], default=0),
        index=score.index,
    )
    turnover = position.diff().abs().fillna(position.abs())
    return position * target - turnover * cost_rate


def optimize_weights_for_sharpe_array(matrix, target_values, candidates, cost_rate):
    candidates = np.asarray(candidates, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    if candidates.size == 0:
        raise ValueError("No weight candidates available.")

    scores = np.clip(50 + 12.5 * (matrix @ candidates.T), 0, 100)
    positions = np.where(scores >= 65, 1.0, np.where(scores <= 35, -1.0, 0.0))
    turnovers = np.abs(np.diff(positions, axis=0, prepend=np.zeros((1, positions.shape[1]))))
    returns = positions * target_values[:, None] - turnovers * cost_rate

    means = returns.mean(axis=0)
    stds = returns.std(axis=0, ddof=1)
    sharpes = np.full(len(candidates), -np.inf)
    valid = stds > 0
    sharpes[valid] = np.sqrt(TRADING_DAYS_PER_YEAR) * means[valid] / stds[valid]

    best_idx = int(np.argmax(sharpes))
    best_sharpe = sharpes[best_idx]
    if not np.isfinite(best_sharpe):
        best_sharpe = np.nan
    return candidates[best_idx], best_sharpe


def optimize_weights_for_sharpe(components, target, candidates, cost_rate):
    best_weights, best_sharpe = optimize_weights_for_sharpe_array(
        components.to_numpy(dtype=float),
        target.to_numpy(dtype=float),
        candidates,
        cost_rate,
    )
    return pd.Series(best_weights, index=components.columns), best_sharpe


def optimize_thresholds_for_sharpe_array(score, target_values, cost_rate):
    best_buy, best_sell, best_sharpe = 65, 35, -np.inf
    for buy_threshold in range(55, 81, 5):
        for sell_threshold in range(20, 46, 5):
            if sell_threshold >= buy_threshold:
                continue
            ret = strategy_returns_from_score_array(score, target_values, buy_threshold, sell_threshold, cost_rate)
            sharpe = annualized_sharpe_array(ret)
            if not pd.isna(sharpe) and sharpe > best_sharpe:
                best_buy, best_sell, best_sharpe = buy_threshold, sell_threshold, sharpe
    return best_buy, best_sell, best_sharpe


def optimize_thresholds_for_sharpe(score, target, cost_rate):
    return optimize_thresholds_for_sharpe_array(
        score.to_numpy(dtype=float),
        target.to_numpy(dtype=float),
        cost_rate,
    )


def build_walk_forward_alpha_signal(components, target, cost_rate):
    frame = components.join(target.rename("target")).dropna()
    component_cols = list(components.columns)
    component_matrix = frame[component_cols].to_numpy(dtype=float)
    target_values = frame["target"].to_numpy(dtype=float)
    candidates = candidate_weight_grid(len(components.columns))
    rows = []
    if len(component_cols) == 3:
        cached_weights = np.array([0.45, 0.35, 0.20], dtype=float)
    else:
        cached_weights = np.full(len(component_cols), 1 / len(component_cols), dtype=float)
    cached_buy, cached_sell = 65, 35
    last_optimized_i = None
    previous_position = 0

    for i in range(ALPHA_MIN_TRAIN, len(frame)):
        if last_optimized_i is None or i - last_optimized_i >= REOPTIMIZE_EVERY_DAYS:
            start = max(0, i - WALK_FORWARD_WINDOW)
            train_matrix = component_matrix[start:i]
            train_target = target_values[start:i]
            cached_weights, _ = optimize_weights_for_sharpe_array(
                train_matrix,
                train_target,
                candidates,
                cost_rate,
            )
            train_score = np.clip(50 + 12.5 * (train_matrix @ cached_weights), 0, 100)
            cached_buy, cached_sell, _ = optimize_thresholds_for_sharpe_array(
                train_score,
                train_target,
                cost_rate,
            )
            last_optimized_i = i

        idx = frame.index[i]
        raw_score = float(component_matrix[i] @ cached_weights)
        alpha_score = float(np.clip(50 + 12.5 * raw_score, 0, 100))
        if alpha_score >= cached_buy:
            position = 1
        elif alpha_score <= cached_sell:
            position = -1
        else:
            position = 0

        turnover = abs(position - previous_position)
        strategy_return = position * target_values[i] - turnover * cost_rate
        row = {
            "Alpha Score": round(alpha_score, 0),
            "Trading Signal": {1: "BUY", -1: "SELL", 0: "NEUTRAL"}[position],
            "Position": position,
            "Buy Threshold": cached_buy,
            "Sell Threshold": cached_sell,
            "Strategy Return": strategy_return,
            "Buy & Hold Return": target_values[i],
        }
        for col, weight in zip(component_cols, cached_weights):
            row[f"Alpha Weight: {col}"] = weight
        rows.append((idx, row))
        previous_position = position

    signals_out = pd.DataFrame([row for _, row in rows], index=[idx for idx, _ in rows])
    return signals_out


def predictive_tests(features, target):
    rows = []
    for col in features.columns:
        frame = pd.concat([features[col], target], axis=1).dropna()
        frame.columns = ["feature", "target"]
        if len(frame) < 30 or frame["feature"].std() == 0:
            rows.append({"Feature": col, "Obs": len(frame), "Corr": np.nan, "Beta": np.nan, "T-stat": np.nan})
            continue
        corr = frame["feature"].corr(frame["target"])
        beta = LinearRegression().fit(frame[["feature"]], frame["target"]).coef_[0]
        if pd.isna(corr) or abs(corr) >= 1:
            t_stat = np.nan
        else:
            t_stat = corr * np.sqrt((len(frame) - 2) / (1 - corr ** 2))
        rows.append({"Feature": col, "Obs": len(frame), "Corr": corr, "Beta": beta, "T-stat": t_stat})
    return pd.DataFrame(rows).set_index("Feature").sort_values("T-stat", key=lambda s: s.abs(), ascending=False)


def active_macro_session_mask(returns):
    """Identify dates where at least one contemporaneous macro market moved."""
    macro_moves = [
        "NASDAQ_ret",
        "DXY_ret",
        "GOLD_ret",
        "OIL_ret",
        "REAL_YIELD_chg",
    ]
    return returns[macro_moves].abs().gt(1e-12).any(axis=1)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Building model signals...")
def build_analysis(data, transaction_cost_bps, collect_timings=False):
    timings = {} if collect_timings else None

    with timed("feature_engineering", timings):
        returns = pd.DataFrame(index=data.index)
        returns["BTC_ret"] = data["BTC"].pct_change()
        returns["NASDAQ_ret"] = data["NASDAQ"].pct_change()
        returns["DXY_ret"] = data["DXY"].pct_change()
        returns["GOLD_ret"] = data["GOLD"].pct_change()
        returns["OIL_ret"] = data["OIL"].pct_change()
        returns["REAL_YIELD_chg"] = data["REAL_YIELD"].diff()
        returns["NASDAQ_lag1"] = returns["NASDAQ_ret"].shift(1)
        returns["DXY_lag1"] = returns["DXY_ret"].shift(1)
        returns["REAL_YIELD_lag1"] = returns["REAL_YIELD_chg"].shift(1)
        returns = returns.dropna()

        if len(returns) < 20:
            raise ValueError("Insufficient data after cleaning.")

        features = [
            "NASDAQ_ret",
            "DXY_ret",
            "GOLD_ret",
            "OIL_ret",
            "REAL_YIELD_chg",
            "NASDAQ_lag1",
            "DXY_lag1",
            "REAL_YIELD_lag1",
        ]
        active_returns = returns.loc[active_macro_session_mask(returns)]
        if len(active_returns) < 20:
            raise ValueError("Insufficient active macro sessions for same-day attribution.")

        X = active_returns[features]
        y = active_returns["BTC_ret"]

    with timed("same_day_models", timings):
        linreg = LinearRegression().fit(X, y)
        coeffs = pd.Series(linreg.coef_, index=features)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
        importance = pd.Series(rf.feature_importances_, index=features)

    latest = X.iloc[-1]
    contrib = coeffs * latest
    pred = float(linreg.predict(latest.to_frame().T)[0])
    actual = y.iloc[-1]

    drivers = []
    for f, v in contrib.sort_values(ascending=False).items():
        if abs(v) > 0.001:
            drivers.append(f"{f} {'up' if v > 0 else 'down'}")
        if len(drivers) >= 3:
            break
    narrative = "BTC move driven by: " + (", ".join(drivers) if drivers else "no dominant factor")

    target_next = returns["BTC_ret"].shift(-1).rename("Next Day BTC Return")
    next_day_frame = returns[features].join(target_next).dropna()
    X_next = next_day_frame[features]
    y_next = next_day_frame["Next Day BTC Return"]
    cost_rate = transaction_cost_bps / 10000

    with timed("walk_forward_model", timings):
        wf_model_pred, wf_model_coefs = walk_forward_linear_model(
            X_next,
            y_next,
            min_train=WALK_FORWARD_MIN_TRAIN,
            train_window=WALK_FORWARD_WINDOW,
        )

    regime_features = pd.DataFrame(index=X_next.index)
    regime_features["NASDAQ_z"] = safe_rolling_z(returns["NASDAQ_ret"]).reindex(X_next.index)
    regime_features["DXY_z"] = safe_rolling_z(returns["DXY_ret"]).reindex(X_next.index)
    regime_features["REAL_YIELD_z"] = safe_rolling_z(returns["REAL_YIELD_chg"]).reindex(X_next.index)
    regime_features["GOLD_z"] = safe_rolling_z(returns["GOLD_ret"]).reindex(X_next.index)

    with timed("walk_forward_regime", timings):
        wf_regime_pred, wf_regime_coefs = walk_forward_linear_model(
            regime_features,
            y_next,
            min_train=WALK_FORWARD_MIN_TRAIN,
            train_window=WALK_FORWARD_WINDOW,
        )
    regime_weights = wf_regime_coefs.apply(normalize_abs_weights, axis=1)
    regime_raw = (regime_features * regime_weights).sum(axis=1, min_count=1)

    regime = pd.DataFrame(index=X_next.index)
    regime["Regime Score"] = (50 + 15 * regime_raw).clip(0, 100)
    regime["Regime"] = regime["Regime Score"].apply(classify_regime)

    btc_momentum_7d = data["BTC"].pct_change(7).reindex(X_next.index).ffill()
    model_prediction_z = safe_rolling_z(wf_model_pred).reindex(X_next.index)
    model_prediction_z[wf_model_pred.reindex(X_next.index).isna()] = np.nan
    regime_prediction_z = safe_rolling_z(wf_regime_pred).reindex(X_next.index)
    regime_prediction_z[wf_regime_pred.reindex(X_next.index).isna()] = np.nan

    alpha_components = pd.DataFrame(index=X_next.index)
    alpha_components["Model Prediction z"] = model_prediction_z
    alpha_components["Regime Prediction z"] = regime_prediction_z
    alpha_components["BTC 7D Momentum z"] = safe_rolling_z(btc_momentum_7d)

    with timed("alpha_signal", timings):
        signals = build_walk_forward_alpha_signal(alpha_components, y_next, cost_rate)

    if len(signals) < 20:
        raise ValueError("Insufficient walk-forward data after calibration.")

    signals["Regime"] = regime["Regime"].reindex(signals.index)
    signals["Regime Score"] = regime["Regime Score"].reindex(signals.index)
    signals["Model Predicted Return"] = wf_model_pred.reindex(signals.index)
    signals["BTC 7D Momentum"] = btc_momentum_7d.reindex(signals.index)
    signals["Strategy Equity"] = (1 + signals["Strategy Return"]).cumprod()
    signals["Buy & Hold Equity"] = (1 + signals["Buy & Hold Return"]).cumprod()
    active_strategy_returns = signals.loc[signals["Position"] != 0, "Strategy Return"]

    backtest_stats = pd.Series(
        {
            "Strategy Sharpe": annualized_sharpe(signals["Strategy Return"]),
            "Buy & Hold Sharpe": annualized_sharpe(signals["Buy & Hold Return"]),
            "Strategy Total Return": signals["Strategy Equity"].iloc[-1] - 1,
            "Buy & Hold Total Return": signals["Buy & Hold Equity"].iloc[-1] - 1,
            "Strategy Max Drawdown": max_drawdown(signals["Strategy Equity"]),
            "Buy & Hold Max Drawdown": max_drawdown(signals["Buy & Hold Equity"]),
            "Signal Win Rate": (active_strategy_returns > 0).mean() if not active_strategy_returns.empty else np.nan,
            "Trading Cost bps": transaction_cost_bps,
            "Latest Buy Threshold": signals["Buy Threshold"].iloc[-1],
            "Latest Sell Threshold": signals["Sell Threshold"].iloc[-1],
        }
    )

    feature_test_frame = pd.concat([regime_features, alpha_components], axis=1)
    predictive_test_results = predictive_tests(feature_test_frame, y_next)
    latest_signal = signals.iloc[-1]
    latest_regime_weights = regime_weights.dropna(how="all").iloc[-1]
    latest_alpha_weights = latest_signal.filter(like="Alpha Weight: ").rename(
        lambda x: x.replace("Alpha Weight: ", "")
    )

    return {
        "returns": returns,
        "coeffs": coeffs,
        "importance": importance,
        "contrib": contrib,
        "attribution_date": latest.name,
        "pred": pred,
        "actual": actual,
        "narrative": narrative,
        "signals": signals,
        "regime_weights": regime_weights,
        "backtest_stats": backtest_stats,
        "predictive_test_results": predictive_test_results,
        "latest_signal": latest_signal,
        "latest_regime_weights": latest_regime_weights,
        "latest_alpha_weights": latest_alpha_weights,
        "timings": timings or {},
    }


try:
    with timed("build_analysis"):
        analysis = build_analysis(data, float(transaction_cost_bps), bool(show_timing))
except ValueError as exc:
    st.error(str(exc))
    st.stop()
    raise SystemExit

returns = analysis["returns"]
coeffs = analysis["coeffs"]
importance = analysis["importance"]
contrib = analysis["contrib"]
attribution_date = analysis["attribution_date"]
pred = analysis["pred"]
actual = analysis["actual"]
narrative = analysis["narrative"]
signals = analysis["signals"]
regime_weights = analysis["regime_weights"]
backtest_stats = analysis["backtest_stats"]
predictive_test_results = analysis["predictive_test_results"]
latest_signal = analysis["latest_signal"]
latest_regime_weights = analysis["latest_regime_weights"]
latest_alpha_weights = analysis["latest_alpha_weights"]

if show_timing:
    for label, elapsed in analysis.get("timings", {}).items():
        st.sidebar.write(f"{label}: {elapsed:.2f}s")

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
sig1, sig2, sig3, sig4, sig5 = st.columns(5)
with sig1:
    st.metric(
        "Regime",
        latest_signal["Regime"],
        delta=f"{latest_signal['Regime Score']:.0f}/100",
    )
with sig2:
    st.metric("BTC Alpha Score", f"{latest_signal['Alpha Score']:.0f}/100")
with sig3:
    st.metric("Latest Evaluated Signal", latest_signal["Trading Signal"])
    st.caption(f"Feature date {pd.Timestamp(latest_signal.name):%Y-%m-%d}")
with sig4:
    sharpe_delta = backtest_stats["Strategy Sharpe"] - backtest_stats["Buy & Hold Sharpe"]
    st.metric(
        "Backtest Sharpe",
        format_metric(backtest_stats["Strategy Sharpe"]),
        delta=f"{sharpe_delta:+.2f} vs HODL" if not pd.isna(sharpe_delta) else None,
    )
with sig5:
    st.metric("10Y Real Yield", f"{data['REAL_YIELD'].iloc[-1]:.2f}%")
    if real_yield_as_of is not None:
        st.caption(f"As of {pd.Timestamp(real_yield_as_of):%Y-%m-%d}")

st.subheader("Regime + BTC Alpha Signal")
st.line_chart(signals[["Alpha Score", "Regime Score"]].tail(365))

coef1, coef2 = st.columns(2)
with coef1:
    st.subheader("Latest Regime Coefficients")
    st.dataframe(latest_regime_weights.astype(float).sort_values(ascending=False).rename("Weight"))
with coef2:
    st.subheader("Latest Alpha Coefficients")
    st.dataframe(latest_alpha_weights.astype(float).sort_values(ascending=False).rename("Weight"))

bt1, bt2, bt3, bt4, bt5 = st.columns(5)
with bt1:
    st.metric("Strategy Sharpe", format_metric(backtest_stats["Strategy Sharpe"]))
with bt2:
    st.metric("Buy & Hold Sharpe", format_metric(backtest_stats["Buy & Hold Sharpe"]))
with bt3:
    st.metric("Strategy Return", format_metric(backtest_stats["Strategy Total Return"], "{:.2%}"))
with bt4:
    st.metric("Strategy Max DD", format_metric(backtest_stats["Strategy Max Drawdown"], "{:.2%}"))
with bt5:
    st.metric("Cost / Trade", f"{transaction_cost_bps:.0f} bps")

st.subheader("Backtest Equity Curve")
fig_bt, ax_bt = plt.subplots()
ax_bt.plot(signals["Strategy Equity"], label="Signal Strategy")
ax_bt.plot(signals["Buy & Hold Equity"], label="Buy & Hold")
ax_bt.axhline(1, color="gray", linewidth=0.5, linestyle="--")
ax_bt.legend()
ax_bt.set_title("Walk-Forward Next-Day Signal Backtest")
st.pyplot(fig_bt)
plt.close(fig_bt)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 BTC Price")
    st.line_chart(data["BTC"])

with col2:
    st.subheader("📊 Rolling Correlation (30D)")
    roll = returns[["BTC_ret","NASDAQ_ret","DXY_ret"]].rolling(30).corr()
    bn = roll.loc[(slice(None),"BTC_ret"),"NASDAQ_ret"].reset_index(level=1, drop=True)
    bd = roll.loc[(slice(None),"BTC_ret"),"DXY_ret"].reset_index(level=1, drop=True)
    fig, ax = plt.subplots()
    ax.plot(bn, label="BTC vs NASDAQ")
    ax.plot(bd, label="BTC vs DXY")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    ax.set_title("Rolling 30D Correlation")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("🧠 Same-Day Attribution Model Coefficients")
st.dataframe(coeffs.sort_values(ascending=False).rename("Coefficient"))

st.subheader("🌲 Feature Importance (Random Forest)")
st.dataframe(importance.sort_values(ascending=False).rename("Importance"))

st.subheader("📌 Driver Attribution (latest active macro session)")
st.caption(
    f"Feature date {pd.Timestamp(attribution_date):%Y-%m-%d}. "
    "Weekends and market holidays are excluded from same-day attribution."
)
st.dataframe(contrib.sort_values(ascending=False).rename("Contribution"))

col3, col4 = st.columns(2)
with col3:
    st.metric("Fitted Same-Day BTC Return", f"{pred:.2%}")
with col4:
    st.metric("Actual BTC Return", f"{actual:.2%}", delta=f"{(actual - pred):.2%} residual")

st.subheader("🧾 Interpretation")
st.write(narrative)

with st.expander("Raw Data"):
    st.dataframe(data.tail(20))
with st.expander("Returns / Features"):
    st.dataframe(returns.tail(20))
with st.expander("Regime / Alpha / Signal History"):
    st.dataframe(
        signals[
            [
                "Regime",
                "Regime Score",
                "Alpha Score",
                "Trading Signal",
                "Model Predicted Return",
                "BTC 7D Momentum",
                "Strategy Return",
            ]
        ].tail(60)
    )
with st.expander("Predictive Tests vs Next-Day BTC Return"):
    st.dataframe(predictive_test_results)
with st.expander("Walk-Forward Coefficient History"):
    alpha_weight_cols = [c for c in signals.columns if c.startswith("Alpha Weight: ")]
    st.write("Regime weights")
    st.dataframe(regime_weights.dropna(how="all").tail(60))
    st.write("Alpha weights")
    st.dataframe(signals[alpha_weight_cols + ["Buy Threshold", "Sell Threshold"]].tail(60))
with st.expander("Backtest Stats"):
    st.dataframe(backtest_stats.rename("Value"))
