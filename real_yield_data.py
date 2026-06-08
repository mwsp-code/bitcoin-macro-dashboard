from io import StringIO
from pathlib import Path

import pandas as pd


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/TextView"
)


def clean_real_yield(raw):
    if raw is None or raw.empty:
        return None

    if "REAL_YIELD" in raw.columns:
        value_column = "REAL_YIELD"
    elif "DFII10" in raw.columns:
        value_column = "DFII10"
    elif len(raw.columns) == 1:
        value_column = raw.columns[0]
    else:
        return None

    values = pd.to_numeric(raw[value_column], errors="coerce").dropna()
    if values.empty:
        return None

    index = pd.to_datetime(values.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    values.index = index.normalize()
    values = values[~values.index.duplicated(keep="last")].sort_index()
    return values.astype(float).rename("REAL_YIELD")


def read_real_yield_file(path):
    path = Path(path)
    if not path.exists():
        return None
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    return clean_real_yield(raw)


def load_fred_real_yield(session):
    response = session.get(FRED_URL, timeout=(5, 15))
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
    series = clean_real_yield(raw)
    if series is None:
        raise ValueError("FRED returned no numeric DFII10 observations")
    return series


def load_treasury_real_yield(session, years):
    pieces = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BTCMacroDashboard/1.0)"}

    for year in sorted(set(int(year) for year in years)):
        response = session.get(
            TREASURY_URL,
            params={
                "type": "daily_treasury_real_yield_curve",
                "field_tdr_date_value": str(year),
            },
            headers=headers,
            timeout=(5, 30),
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text), match="10 YR")
        if not tables:
            continue

        frame = tables[0]
        if "Date" not in frame.columns or "10 YR" not in frame.columns:
            continue

        piece = pd.Series(
            pd.to_numeric(frame["10 YR"], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame["Date"], errors="coerce"),
            name="REAL_YIELD",
        ).dropna()
        if not piece.empty:
            pieces.append(piece)

    if not pieces:
        raise ValueError("U.S. Treasury returned no 10-year real-yield rows")

    raw = pd.concat(pieces).to_frame()
    series = clean_real_yield(raw)
    if series is None:
        raise ValueError("U.S. Treasury real-yield rows were not numeric")
    return series


def merge_real_yield_history(*series_items):
    available = [series for series in series_items if series is not None and not series.empty]
    if not available:
        return None
    merged = pd.concat(available).sort_index()
    return merged[~merged.index.duplicated(keep="last")].rename("REAL_YIELD")


def write_runtime_cache(series, cache_file, errors):
    try:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        series.to_frame().to_csv(cache_file)
    except Exception as exc:
        errors.append(f"cache write: {type(exc).__name__}: {exc}")


def load_real_yield(session, cache_file, seed_file, now=None):
    errors = []
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)

    runtime_cache = None
    seed = None
    for path, label in ((cache_file, "runtime cache"), (seed_file, "bundled seed")):
        try:
            series = read_real_yield_file(path)
            if label == "runtime cache":
                runtime_cache = series
            else:
                seed = series
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    fallback = merge_real_yield_history(seed, runtime_cache)

    try:
        fred = load_fred_real_yield(session)
        combined = merge_real_yield_history(fallback, fred)
        write_runtime_cache(combined, cache_file, errors)
        return combined, "live (FRED)", combined.index[-1], errors
    except Exception as exc:
        errors.append(f"FRED: {type(exc).__name__}: {exc}")

    try:
        start_year = fallback.index[-1].year if fallback is not None else now.year
        treasury = load_treasury_real_yield(session, {start_year, now.year})
        combined = merge_real_yield_history(fallback, treasury)
        write_runtime_cache(combined, cache_file, errors)
        return combined, "live (U.S. Treasury fallback)", combined.index[-1], errors
    except Exception as exc:
        errors.append(f"U.S. Treasury: {type(exc).__name__}: {exc}")

    if fallback is not None:
        return fallback, "stale bundled/cache fallback", fallback.index[-1], errors

    return None, "failed", None, errors
