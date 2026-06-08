from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import json
import os
import socket
import time

import numpy as np
import pandas as pd
import requests

from .config import (
    COMMON_LOCAL_PROXY_PORTS,
    DataPaths,
    NASDAQ_FALLBACK_TICKERS,
    REQUIRED_MARKET_COLUMNS,
    SCHEMA_VERSION,
    YFINANCE_TICKERS,
)


CACHE_MAX_AGE_HOURS = 23
BINANCE_START = "2017-01-01"
BTC_COLUMNS = [
    "BTC",
    "BTC_OPEN",
    "BTC_HIGH",
    "BTC_LOW",
    "BTC_VOLUME",
    "BTC_QUOTE_VOLUME",
    "BTC_TRADES",
    "BTC_TAKER_BUY_VOLUME",
]


@dataclass
class MarketDataBundle:
    data: pd.DataFrame
    status: dict
    timestamps: dict
    generated_at_utc: pd.Timestamp
    btc_source: str
    historical_mode: bool = False


def utc_now():
    return pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)


def normalize_daily_index(index):
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True, errors="coerce"))
    return idx.tz_convert(None).normalize()


def normalize_proxy_url(value):
    if value in (None, "", "0", "none", "None", "false", "False"):
        return None
    value = str(value).strip()
    return value if "://" in value else f"http://{value}"


def local_port_is_open(port, timeout=0.08):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_proxy(settings=None):
    settings = settings or {}
    explicit_url = normalize_proxy_url(
        settings.get("BTC_PROXY_URL") or os.environ.get("BTC_PROXY_URL")
    )
    if explicit_url:
        return explicit_url, "BTC_PROXY_URL"

    explicit_port = (
        settings.get("BTC_PROXY_PORT")
        or settings.get("PROXY_PORT")
        or os.environ.get("BTC_PROXY_PORT")
        or os.environ.get("PROXY_PORT")
    )
    if explicit_port not in (None, ""):
        try:
            return f"http://127.0.0.1:{int(explicit_port)}", "BTC_PROXY_PORT"
        except (TypeError, ValueError):
            pass

    for name in (
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        proxy = normalize_proxy_url(os.environ.get(name))
        if proxy:
            return proxy, name

    auto_detect = settings.get(
        "BTC_PROXY_AUTO_DETECT", os.environ.get("BTC_PROXY_AUTO_DETECT")
    )
    if auto_detect not in ("0", "false", "False", "no", "No"):
        for port in COMMON_LOCAL_PROXY_PORTS:
            if local_port_is_open(port):
                return f"http://127.0.0.1:{port}", f"auto-detected port {port}"
    return None, "direct connection"


def build_session(proxy_url=None):
    session = requests.Session()
    session.trust_env = False
    if proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})
    return session


def _empty_btc_frame():
    return pd.DataFrame(columns=BTC_COLUMNS, dtype=float)


def parse_binance_klines(rows, now_utc=None):
    if not rows:
        return _empty_btc_frame()
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns[: len(rows[0])])
    now_timestamp = pd.Timestamp(now_utc if now_utc is not None else utc_now())
    if now_timestamp.tzinfo is None:
        now_timestamp = now_timestamp.tz_localize("UTC")
    else:
        now_timestamp = now_timestamp.tz_convert("UTC")
    now_ms = int(now_timestamp.value // 1_000_000)
    frame["close_time"] = pd.to_numeric(frame["close_time"], errors="coerce")
    frame = frame.loc[frame["close_time"] < now_ms].copy()
    if frame.empty:
        return _empty_btc_frame()
    frame.index = normalize_daily_index(
        pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    )
    mapping = {
        "close": "BTC",
        "open": "BTC_OPEN",
        "high": "BTC_HIGH",
        "low": "BTC_LOW",
        "volume": "BTC_VOLUME",
        "quote_volume": "BTC_QUOTE_VOLUME",
        "trades": "BTC_TRADES",
        "taker_buy_base": "BTC_TAKER_BUY_VOLUME",
    }
    output = frame[list(mapping)].rename(columns=mapping)
    output = output.apply(pd.to_numeric, errors="coerce")
    return output[~output.index.duplicated(keep="last")].sort_index()


def load_btc_binance(session, start_date=BINANCE_START, max_pages=10):
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    rows = []
    for _ in range(max_pages):
        response = session.get(
            url,
            params={
                "symbol": "BTCUSDT",
                "interval": "1d",
                "limit": 1000,
                "startTime": start_ms,
            },
            timeout=15,
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        next_start = int(page[-1][0]) + 86_400_000
        if next_start <= start_ms or len(page) < 1000:
            break
        start_ms = next_start
    frame = parse_binance_klines(rows)
    if frame.empty:
        raise ValueError("Binance returned no completed daily candles")
    return frame


def load_btc_huobi(session):
    response = session.get(
        "https://api.huobi.pro/market/history/kline",
        params={"symbol": "btcusdt", "period": "1day", "size": 2000},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok" or not payload.get("data"):
        raise ValueError("HTX returned no daily candles")
    frame = pd.DataFrame(payload["data"])
    frame.index = normalize_daily_index(
        pd.to_datetime(frame["id"], unit="s", utc=True)
    )
    frame = frame.loc[frame.index < utc_now().normalize()].copy()
    mapping = {
        "close": "BTC",
        "open": "BTC_OPEN",
        "high": "BTC_HIGH",
        "low": "BTC_LOW",
        "amount": "BTC_VOLUME",
        "vol": "BTC_QUOTE_VOLUME",
        "count": "BTC_TRADES",
    }
    output = frame[list(mapping)].rename(columns=mapping).apply(
        pd.to_numeric, errors="coerce"
    )
    output["BTC_TAKER_BUY_VOLUME"] = np.nan
    return output[BTC_COLUMNS].sort_index()


def load_btc_okx(session, max_pages=25):
    url = "https://www.okx.com/api/v5/market/history-candles"
    rows = []
    after = None
    for _ in range(max_pages):
        params = {"instId": "BTC-USDT", "bar": "1Dutc", "limit": "100"}
        if after is not None:
            params["after"] = after
        response = session.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("data", [])
        if payload.get("code") != "0" or not page:
            break
        rows.extend(page)
        next_after = page[-1][0]
        if next_after == after or len(page) < 100:
            break
        after = next_after
    if not rows:
        raise ValueError("OKX returned no daily candles")
    frame = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_currency",
            "quote_volume",
            "confirm",
        ],
    )
    frame = frame.loc[frame["confirm"].astype(str) == "1"].copy()
    frame.index = normalize_daily_index(
        pd.to_datetime(frame["time"].astype(float), unit="ms", utc=True)
    )
    mapping = {
        "close": "BTC",
        "open": "BTC_OPEN",
        "high": "BTC_HIGH",
        "low": "BTC_LOW",
        "volume": "BTC_VOLUME",
        "quote_volume": "BTC_QUOTE_VOLUME",
    }
    output = frame[list(mapping)].rename(columns=mapping).apply(
        pd.to_numeric, errors="coerce"
    )
    output["BTC_TRADES"] = np.nan
    output["BTC_TAKER_BUY_VOLUME"] = np.nan
    return output[BTC_COLUMNS].sort_index()


def load_btc_coingecko(session):
    response = session.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency": "usd", "days": "max", "interval": "daily"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    prices = pd.DataFrame(payload.get("prices", []), columns=["time", "BTC"])
    volumes = pd.DataFrame(
        payload.get("total_volumes", []), columns=["time", "BTC_QUOTE_VOLUME"]
    )
    if prices.empty:
        raise ValueError("CoinGecko returned no prices")
    frame = prices.merge(volumes, on="time", how="left")
    frame.index = normalize_daily_index(
        pd.to_datetime(frame["time"], unit="ms", utc=True)
    )
    frame = frame.loc[frame.index < utc_now().normalize()]
    for column in ("BTC_OPEN", "BTC_HIGH", "BTC_LOW"):
        frame[column] = frame["BTC"]
    frame["BTC_VOLUME"] = np.nan
    frame["BTC_TRADES"] = np.nan
    frame["BTC_TAKER_BUY_VOLUME"] = np.nan
    return frame[BTC_COLUMNS].apply(pd.to_numeric, errors="coerce").sort_index()


def load_btc(session):
    errors = {}
    loaders = [
        ("binance", load_btc_binance),
        ("htx", load_btc_huobi),
        ("okx", load_btc_okx),
        ("coingecko", load_btc_coingecko),
    ]
    for name, loader in loaders:
        try:
            frame = loader(session)
            if len(frame) < 400:
                raise ValueError(f"only {len(frame)} completed observations")
            return frame, name, errors
        except Exception as exc:
            errors[name.upper()] = f"{type(exc).__name__}: {exc}"
    return None, None, errors


def _extract_yfinance_close(raw, ticker):
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
    return close[~close.index.duplicated(keep="last")].astype(float)


def load_macro_yfinance(start_date, proxy_url=None):
    try:
        import yfinance as yf
    except ImportError:
        return {}, {name: "yfinance unavailable" for _, name in YFINANCE_TICKERS}, {}

    old_http = os.environ.get("HTTP_PROXY")
    old_https = os.environ.get("HTTPS_PROXY")
    try:
        if proxy_url:
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
        raw = yf.download(
            [ticker for ticker, _ in YFINANCE_TICKERS],
            start=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            threads=True,
            progress=False,
            timeout=15,
            group_by="ticker",
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {}, {name: message for _, name in YFINANCE_TICKERS}, {}
    finally:
        if old_http is None:
            os.environ.pop("HTTP_PROXY", None)
        else:
            os.environ["HTTP_PROXY"] = old_http
        if old_https is None:
            os.environ.pop("HTTPS_PROXY", None)
        else:
            os.environ["HTTPS_PROXY"] = old_https

    series_by_name = {}
    status = {}
    timestamps = {}
    for ticker, name in YFINANCE_TICKERS:
        close = _extract_yfinance_close(raw, ticker)
        if close is None:
            status[name] = "empty Yahoo response"
            continue
        series_by_name[name] = close.rename(name)
        status[name] = f"live (Yahoo {ticker})"
        timestamps[name] = close.index[-1]
    return series_by_name, status, timestamps


def load_macro_nasdaq(session, start_date):
    series_by_name = {}
    status = {}
    timestamps = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/market-activity/",
    }
    for ticker, name in NASDAQ_FALLBACK_TICKERS:
        try:
            response = session.get(
                f"https://api.nasdaq.com/api/quote/{ticker}/historical",
                params={
                    "assetclass": "etf",
                    "fromdate": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                    "todate": utc_now().strftime("%Y-%m-%d"),
                    "limit": "5000",
                },
                headers=headers,
                timeout=25,
            )
            response.raise_for_status()
            rows = (
                response.json()
                .get("data", {})
                .get("tradesTable", {})
                .get("rows", [])
            )
            frame = pd.DataFrame(rows)
            if frame.empty:
                raise ValueError("no historical rows")
            dates = normalize_daily_index(frame["date"])
            closes = pd.to_numeric(
                frame["close"].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            )
            series = pd.Series(closes.to_numpy(), index=dates, name=name).dropna()
            series = series[~series.index.duplicated(keep="last")].sort_index()
            series_by_name[name] = series
            proxy = ticker if ticker == "QQQ" else f"{ticker} proxy"
            status[name] = f"live (Nasdaq {proxy})"
            timestamps[name] = series.index[-1]
        except Exception as exc:
            status[name] = f"failed Nasdaq {ticker}: {type(exc).__name__}: {exc}"
    return series_by_name, status, timestamps


def clean_real_yield(raw):
    if raw is None or raw.empty:
        return None
    frame = raw.copy()
    if "REAL_YIELD" in frame.columns:
        column = "REAL_YIELD"
    elif "DFII10" in frame.columns:
        column = "DFII10"
    elif len(frame.columns) == 1:
        column = frame.columns[0]
    else:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    series.index = normalize_daily_index(series.index)
    return (
        series[~series.index.duplicated(keep="last")]
        .sort_index()
        .astype(float)
        .rename("REAL_YIELD")
    )


def read_real_yield_fallback(paths):
    candidates = []
    errors = []
    for path, label in (
        (paths.real_yield_cache_file, "real-yield cache"),
        (paths.cache_file, "complete-data cache"),
        (paths.real_yield_seed_file, "bundled official seed"),
    ):
        if not path.exists():
            continue
        try:
            series = clean_real_yield(
                pd.read_csv(path, index_col=0, parse_dates=True)
            )
            if series is not None:
                candidates.append((series.index[-1], series, label))
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    if not candidates:
        return None, None, errors
    _, series, label = max(candidates, key=lambda item: item[0])
    return series, label, errors


def load_treasury_real_yield(session, start_year):
    pieces = []
    for year in range(max(2003, int(start_year)), utc_now().year + 1):
        response = session.get(
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView",
            params={
                "type": "daily_treasury_real_yield_curve",
                "field_tdr_date_value": str(year),
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=40,
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text), match="10 YR")
        if tables and "Date" in tables[0] and "10 YR" in tables[0]:
            piece = pd.Series(
                pd.to_numeric(tables[0]["10 YR"], errors="coerce").to_numpy(),
                index=pd.to_datetime(tables[0]["Date"], errors="coerce"),
                name="REAL_YIELD",
            ).dropna()
            pieces.append(piece)
    if not pieces:
        return None
    series = pd.concat(pieces)
    series.index = normalize_daily_index(series.index)
    return series[~series.index.duplicated(keep="last")].sort_index().astype(float)


def load_real_yield(session, paths):
    errors = []
    try:
        response = session.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10",
            timeout=(5, 15),
        )
        response.raise_for_status()
        series = clean_real_yield(
            pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
        )
        if series is None:
            raise ValueError("FRED returned no numeric observations")
        series.to_frame().to_csv(paths.real_yield_cache_file)
        return series, "live (FRED DFII10)", series.index[-1], errors
    except Exception as exc:
        errors.append(f"FRED: {type(exc).__name__}: {exc}")

    fallback, label, fallback_errors = read_real_yield_fallback(paths)
    errors.extend(fallback_errors)
    try:
        start_year = 2020 if fallback is None else fallback.index[-1].year
        treasury = load_treasury_real_yield(session, start_year)
        if treasury is not None:
            if fallback is not None:
                treasury = pd.concat([fallback, treasury])
                treasury = treasury[~treasury.index.duplicated(keep="last")]
            treasury = treasury.sort_index()
            treasury.to_frame().to_csv(paths.real_yield_cache_file)
            return (
                treasury,
                "live (U.S. Treasury fallback)",
                treasury.index[-1],
                errors,
            )
    except Exception as exc:
        errors.append(f"Treasury: {type(exc).__name__}: {exc}")
    if fallback is not None:
        return fallback, f"stale {label}", fallback.index[-1], errors
    return None, "failed", None, errors


def _metadata_is_current(paths):
    if not paths.cache_file.exists() or not paths.cache_metadata_file.exists():
        return False
    try:
        metadata = json.loads(paths.cache_metadata_file.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            return False
        age_hours = (time.time() - paths.cache_file.stat().st_mtime) / 3600
        if age_hours >= CACHE_MAX_AGE_HOURS:
            return False
        data = pd.read_csv(paths.cache_file, index_col=0, parse_dates=True)
        latest = pd.Timestamp(data.index[-1]).normalize()
        return (utc_now().normalize() - latest).days <= 2
    except Exception:
        return False


def _read_cache(paths, historical_mode=True):
    data = pd.read_csv(paths.cache_file, index_col=0, parse_dates=True)
    metadata = {}
    if paths.cache_metadata_file.exists():
        try:
            metadata = json.loads(
                paths.cache_metadata_file.read_text(encoding="utf-8")
            )
        except Exception:
            metadata = {}
    timestamps = {
        key: pd.Timestamp(value)
        for key, value in metadata.get("timestamps", {}).items()
    }
    return MarketDataBundle(
        data=data,
        status=metadata.get("status", {}),
        timestamps=timestamps,
        generated_at_utc=pd.Timestamp(
            metadata.get("generated_at_utc", utc_now())
        ),
        btc_source=metadata.get("btc_source", "cache"),
        historical_mode=historical_mode,
    )


def _write_cache(paths, bundle):
    bundle.data.to_csv(paths.cache_file)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": str(bundle.generated_at_utc),
        "btc_source": bundle.btc_source,
        "status": bundle.status,
        "timestamps": {
            key: str(pd.Timestamp(value))
            for key, value in bundle.timestamps.items()
            if value is not None
        },
    }
    paths.cache_metadata_file.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _align_observed_series(frame, series, name):
    raw = series.reindex(frame.index)
    frame[f"{name}_OBSERVED"] = raw.notna()
    frame[name] = raw.ffill()
    observation_dates = pd.Series(
        pd.NaT, index=frame.index, dtype="datetime64[ns]"
    )
    observation_dates.loc[raw.notna()] = frame.index[raw.notna()]
    observation_dates = observation_dates.ffill()
    frame[f"{name}_AGE_DAYS"] = (
        frame.index.to_series() - observation_dates
    ).dt.days.astype(float)


def load_market_data(base_dir, proxy_url=None, force_refresh=False):
    paths = DataPaths(base_dir)
    if not force_refresh and _metadata_is_current(paths):
        bundle = _read_cache(paths, historical_mode=False)
        bundle.status["_info"] = "Loaded current local cache."
        return bundle

    session = build_session(proxy_url)
    btc, btc_source, btc_errors = load_btc(session)
    if btc is None:
        if paths.cache_file.exists():
            bundle = _read_cache(paths)
            bundle.status["BTC"] = "stale cache; all live sources failed"
            bundle.status["_btc_errors"] = btc_errors
            bundle.historical_mode = True
            return bundle
        raise RuntimeError(f"All BTC sources failed: {btc_errors}")

    status = {"BTC": f"live ({btc_source}; completed UTC bars)"}
    timestamps = {"BTC": btc.index[-1]}
    if btc_errors:
        status["_btc_errors"] = btc_errors

    macro, macro_status, macro_timestamps = load_macro_yfinance(
        btc.index.min(), proxy_url
    )
    expected = {name for _, name in YFINANCE_TICKERS}
    if set(macro) != expected:
        yahoo_errors = macro_status.copy()
        fallback, fallback_status, fallback_timestamps = load_macro_nasdaq(
            session, btc.index.min()
        )
        if set(fallback) == expected:
            macro = fallback
            macro_status = fallback_status
            macro_timestamps = fallback_timestamps
            status["_macro_info"] = (
                "Yahoo unavailable; using one consistent QQQ/UUP/GLD/USO "
                "history without instrument splicing."
            )
            status["_macro_errors"] = yahoo_errors

    real_yield, real_yield_status, real_yield_time, real_yield_errors = (
        load_real_yield(session, paths)
    )
    if real_yield_errors:
        status["_real_yield_errors"] = real_yield_errors
    status.update(macro_status)
    status["REAL_YIELD"] = real_yield_status
    timestamps.update(macro_timestamps)
    if real_yield_time is not None:
        timestamps["REAL_YIELD"] = real_yield_time

    missing = expected.difference(macro)
    if real_yield is None:
        missing.add("REAL_YIELD")
    if missing:
        if paths.cache_file.exists():
            bundle = _read_cache(paths)
            bundle.status["_info"] = (
                f"Live refresh missing {sorted(missing)}; using complete cache."
            )
            bundle.historical_mode = True
            return bundle
        raise RuntimeError(f"Missing live datasets: {sorted(missing)}")

    data = btc.copy().sort_index()
    for name, series in macro.items():
        _align_observed_series(data, series, name)
    _align_observed_series(data, real_yield, "REAL_YIELD")
    data = data.dropna(subset=REQUIRED_MARKET_COLUMNS)
    if data.empty:
        raise RuntimeError("No overlapping complete market observations")

    bundle = MarketDataBundle(
        data=data,
        status=status,
        timestamps=timestamps,
        generated_at_utc=utc_now(),
        btc_source=btc_source,
        historical_mode=False,
    )
    _write_cache(paths, bundle)
    return bundle
