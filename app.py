import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from itertools import product
import requests
import os
import sys
import time

CACHE_FILE    = "backup_data.csv"
CACHE_MAX_AGE = 23   # hours
TRADING_DAYS_PER_YEAR = 365
WALK_FORWARD_MIN_TRAIN = 365
WALK_FORWARD_WINDOW = 730
ALPHA_MIN_TRAIN = 180
REOPTIMIZE_EVERY_DAYS = 7
REQUIRED_COLS = ["BTC", "NASDAQ", "DXY", "GOLD", "OIL", "REAL_YIELD"]

# ── PROXY CONFIG (required for mainland China) ───────────────────────────────
# All crypto APIs are blocked by GFW. Requests must go through your local
# VPN proxy (Clash / V2Ray / Shadowsocks etc).
# Set PROXY_PORT to match your VPN tool:
#   Clash:       7890  (default)
#   V2Ray:       10809 (default)
#   Shadowsocks: 1080  (default)
# Set PROXY_PORT = None to attempt direct connection (will timeout in China).
PROXY_PORT = 7890

SESSION = requests.Session()
SESSION.trust_env = False   # ignore broken system proxy env vars

if PROXY_PORT:
    proxy_url = f"http://127.0.0.1:{PROXY_PORT}"
    SESSION.proxies.update({"http": proxy_url, "https": proxy_url})

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
        cached = pd.read_csv(CACHE_FILE, nrows=5)
        return not cached.empty and all(c in cached.columns for c in REQUIRED_COLS)
    except Exception:
        return False

def load_cache():
    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    status = {col: "cache" for col in df.columns}
    timestamps = {col: df.index[-1] for col in df.columns}
    return df, status, timestamps


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


def load_macro_yfinance(ticker, name):
    """
    Download a macro series via yfinance.
    New yfinance uses curl_cffi internally and no longer accepts a
    requests.Session. Instead we set HTTP_PROXY / HTTPS_PROXY env vars
    before the call — curl_cffi picks these up automatically — then
    restore the original values so SESSION behaviour is unaffected.
    """
    if not YFINANCE_AVAILABLE:
        return None, "yfinance not installed — run: pip install yfinance", None
    try:
        # Temporarily set proxy env vars for curl_cffi
        proxy_url = f"http://127.0.0.1:{PROXY_PORT}" if PROXY_PORT else ""
        old_http  = os.environ.get("HTTP_PROXY", "")
        old_https = os.environ.get("HTTPS_PROXY", "")
        if proxy_url:
            os.environ["HTTP_PROXY"]  = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url

        try:
            tk = yf.Ticker(ticker)   # no session= — let yfinance handle it
            df = tk.history(period="max", interval="1d", auto_adjust=True)
        finally:
            # Always restore original env vars
            os.environ["HTTP_PROXY"]  = old_http
            os.environ["HTTPS_PROXY"] = old_https

        if df.empty:
            return None, "empty response", None
        close = df["Close"].squeeze()
        close.index = normalize_daily_index(close.index)
        close = close[~close.index.duplicated(keep="last")]
        return close.rename(name).astype(float), "live", close.index[-1]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", None

# ─────────────────────────────────────────────
# MAIN DATA LOADER
# ─────────────────────────────────────────────

@st.cache_data
def load_data(force_refresh=False):

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
            data = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
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

    # ── MACRO (yfinance via proxy) ───────────
    for ticker, name in [("QQQ","NASDAQ"), ("DX-Y.NYB","DXY"), ("GC=F","GOLD"), ("CL=F","OIL")]:
        series, stat, ts = load_macro_yfinance(ticker, name)
        status[name] = stat
        if series is not None:
            data[name] = series
            timestamps[name] = ts

    # ── FRED REAL YIELD ──────────────────────
    try:
        ry = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10", index_col=0, parse_dates=True)
        ry.index = pd.to_datetime(ry.index).tz_localize(None).normalize()

        ry.columns = ["REAL_YIELD"]
        ry.index = pd.to_datetime(ry.index).normalize()
        ry = ry[ry["REAL_YIELD"] != "."]
        ry["REAL_YIELD"] = ry["REAL_YIELD"].astype(float)
        if "REAL_YIELD" in data.columns:
            data = data.drop(columns=["REAL_YIELD"])
        data = data.join(ry, how="left")
        status["REAL_YIELD"] = "live"
        timestamps["REAL_YIELD"] = ry.index[-1]
    except Exception as e:
        status["REAL_YIELD"] = f"failed ({type(e).__name__}: {e})"

    missing_live = [c for c in REQUIRED_COLS if c not in data.columns]
    if missing_live:
        cached_data, _, cached_timestamps = load_cache_if_complete()
        if cached_data is not None:
            status["_info"] = f"live refresh incomplete ({missing_live}); showing last complete cache."
            return cached_data, status, {**cached_timestamps, **timestamps}

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
            return cached_data, status, {**cached_timestamps, **timestamps}

    # Optional: show coverage so you can see which series is too sparse
    st.sidebar.write("### 📊 Coverage")
    for c in REQUIRED_COLS:
        if c in data.columns:
            st.sidebar.write(f"{c}: {data[c].notna().sum()} rows")

    if not data.empty:
        data.to_csv(CACHE_FILE)

    return data, status, timestamps

data, status, timestamps = load_data(force_refresh=manual_refresh)
st.sidebar.write("### 🧪 Index Debug")

for col in data.columns:
    st.sidebar.write(
        f"{col}: {data[col].index.min()} → {data[col].index.max()} | rows={len(data[col].dropna())}"
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.write("### 📡 Data Status")
for k, v in status.items():
    if k == "_info":
        st.sidebar.info(f"ℹ️ {v}")
    elif k == "_btc_errors":
        with st.sidebar.expander("🔍 BTC source errors"):
            st.text(v)
    else:
        icon = "✅" if any(x in str(v) for x in ["live", "cache"]) else "❌"
        st.sidebar.write(f"{icon} {k}: {v}")

st.sidebar.write("### 🕒 Freshness")
for k, t in timestamps.items():
    st.sidebar.write(f"{k}: {t}")
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
    predictions = pd.Series(np.nan, index=features.index, name="Walk-Forward Prediction")
    coefs = pd.DataFrame(np.nan, index=features.index, columns=features.columns)

    for i in range(min_train, len(frame)):
        start = max(0, i - train_window) if train_window else 0
        x_train = x.iloc[start:i]
        y_train = y_target.iloc[start:i]
        model = LinearRegression().fit(x_train, y_train)
        idx = x.index[i]
        predictions.loc[idx] = model.predict(x.iloc[[i]])[0]
        coefs.loc[idx] = model.coef_

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


def optimize_weights_for_sharpe(components, target, candidates, cost_rate):
    best_weights = candidates[0]
    best_sharpe = -np.inf
    matrix = components.to_numpy(dtype=float)
    for weights in candidates:
        score = pd.Series((50 + 12.5 * (matrix @ weights)).clip(0, 100), index=components.index)
        ret = strategy_returns_from_score(score, target, 65, 35, cost_rate)
        sharpe = annualized_sharpe(ret)
        if not pd.isna(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
    return pd.Series(best_weights, index=components.columns), best_sharpe


def optimize_thresholds_for_sharpe(score, target, cost_rate):
    best_buy, best_sell, best_sharpe = 65, 35, -np.inf
    for buy_threshold in range(55, 81, 5):
        for sell_threshold in range(20, 46, 5):
            if sell_threshold >= buy_threshold:
                continue
            ret = strategy_returns_from_score(score, target, buy_threshold, sell_threshold, cost_rate)
            sharpe = annualized_sharpe(ret)
            if not pd.isna(sharpe) and sharpe > best_sharpe:
                best_buy, best_sell, best_sharpe = buy_threshold, sell_threshold, sharpe
    return best_buy, best_sell, best_sharpe


def build_walk_forward_alpha_signal(components, target, cost_rate):
    frame = components.join(target.rename("target")).dropna()
    candidates = candidate_weight_grid(len(components.columns))
    rows = []
    cached_weights = pd.Series([0.45, 0.35, 0.20], index=components.columns)
    cached_buy, cached_sell = 65, 35
    last_optimized_i = None
    previous_position = 0

    for i in range(ALPHA_MIN_TRAIN, len(frame)):
        if last_optimized_i is None or i - last_optimized_i >= REOPTIMIZE_EVERY_DAYS:
            start = max(0, i - WALK_FORWARD_WINDOW)
            train = frame.iloc[start:i]
            cached_weights, _ = optimize_weights_for_sharpe(
                train[components.columns],
                train["target"],
                candidates,
                cost_rate,
            )
            train_score = pd.Series(
                (50 + 12.5 * (train[components.columns].to_numpy(dtype=float) @ cached_weights.to_numpy())).clip(0, 100),
                index=train.index,
            )
            cached_buy, cached_sell, _ = optimize_thresholds_for_sharpe(
                train_score,
                train["target"],
                cost_rate,
            )
            last_optimized_i = i

        idx = frame.index[i]
        raw_score = float(frame.loc[idx, components.columns].to_numpy(dtype=float) @ cached_weights.to_numpy())
        alpha_score = float(np.clip(50 + 12.5 * raw_score, 0, 100))
        if alpha_score >= cached_buy:
            position = 1
        elif alpha_score <= cached_sell:
            position = -1
        else:
            position = 0

        turnover = abs(position - previous_position)
        strategy_return = position * frame.loc[idx, "target"] - turnover * cost_rate
        row = {
            "Alpha Score": round(alpha_score, 0),
            "Trading Signal": {1: "BUY", -1: "SELL", 0: "NEUTRAL"}[position],
            "Position": position,
            "Buy Threshold": cached_buy,
            "Sell Threshold": cached_sell,
            "Strategy Return": strategy_return,
            "Buy & Hold Return": frame.loc[idx, "target"],
        }
        for col, weight in cached_weights.items():
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

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
returns = pd.DataFrame(index=data.index)
returns["BTC_ret"]        = data["BTC"].pct_change()
returns["NASDAQ_ret"]     = data["NASDAQ"].pct_change()
returns["DXY_ret"]        = data["DXY"].pct_change()
returns["GOLD_ret"]       = data["GOLD"].pct_change()
returns["OIL_ret"]        = data["OIL"].pct_change()
returns["REAL_YIELD_chg"] = data["REAL_YIELD"].diff()
returns["NASDAQ_lag1"]    = returns["NASDAQ_ret"].shift(1)
returns["DXY_lag1"]       = returns["DXY_ret"].shift(1)
returns["REAL_YIELD_lag1"]= returns["REAL_YIELD_chg"].shift(1)
returns = returns.dropna()

if len(returns) < 20:
    st.error("Insufficient data after cleaning.")
    st.stop()
    raise SystemExit

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
features = ["NASDAQ_ret","DXY_ret","GOLD_ret","OIL_ret",
            "REAL_YIELD_chg","NASDAQ_lag1","DXY_lag1","REAL_YIELD_lag1"]
X = returns[features]
y = returns["BTC_ret"]

linreg    = LinearRegression().fit(X, y)
coeffs    = pd.Series(linreg.coef_, index=features)
model_pred_series = pd.Series(linreg.predict(X), index=X.index, name="Model Predicted BTC Return")
rf        = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
importance= pd.Series(rf.feature_importances_, index=features)

# ─────────────────────────────────────────────
# ATTRIBUTION
# ─────────────────────────────────────────────
latest  = X.iloc[-1]
contrib = coeffs * latest
pred    = contrib.sum()
actual  = y.iloc[-1]

drivers = []
for f, v in contrib.sort_values(ascending=False).items():
    if abs(v) > 0.001:
        drivers.append(f"{f} {'up' if v > 0 else 'down'}")
    if len(drivers) >= 3:
        break
narrative = "BTC move driven by: " + (", ".join(drivers) if drivers else "no dominant factor")

# REGIME, ALPHA SIGNAL, AND BACKTEST
target_next = returns["BTC_ret"].shift(-1).rename("Next Day BTC Return")
next_day_frame = returns[features].join(target_next).dropna()
X_next = next_day_frame[features]
y_next = next_day_frame["Next Day BTC Return"]
cost_rate = transaction_cost_bps / 10000

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

signals = build_walk_forward_alpha_signal(alpha_components, y_next, cost_rate)

if len(signals) < 20:
    st.error("Insufficient walk-forward data after calibration.")
    st.stop()
    raise SystemExit

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

feature_test_frame = pd.concat(
    [
        regime_features,
        alpha_components,
    ],
    axis=1,
)
predictive_test_results = predictive_tests(feature_test_frame, y_next)
latest_signal = signals.iloc[-1]
latest_regime_weights = regime_weights.dropna(how="all").iloc[-1]
latest_alpha_weights = latest_signal.filter(like="Alpha Weight: ").rename(
    lambda x: x.replace("Alpha Weight: ", "")
)

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
sig1, sig2, sig3, sig4 = st.columns(4)
with sig1:
    st.metric(
        "Regime",
        latest_signal["Regime"],
        delta=f"{latest_signal['Regime Score']:.0f}/100",
    )
with sig2:
    st.metric("BTC Alpha Score", f"{latest_signal['Alpha Score']:.0f}/100")
with sig3:
    st.metric("Trading Signal", latest_signal["Trading Signal"])
with sig4:
    sharpe_delta = backtest_stats["Strategy Sharpe"] - backtest_stats["Buy & Hold Sharpe"]
    st.metric(
        "Backtest Sharpe",
        format_metric(backtest_stats["Strategy Sharpe"]),
        delta=f"{sharpe_delta:+.2f} vs HODL" if not pd.isna(sharpe_delta) else None,
    )

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

st.subheader("📌 Driver Attribution (today)")
st.dataframe(contrib.sort_values(ascending=False).rename("Contribution"))

col3, col4 = st.columns(2)
with col3:
    st.metric("Predicted BTC Return", f"{pred:.2%}")
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
