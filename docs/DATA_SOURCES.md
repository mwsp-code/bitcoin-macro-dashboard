# Data Sources

## Source Priority

| Dataset | Preferred source | Fallback | Notes |
| --- | --- | --- | --- |
| Bitcoin | HTX | OKX, Binance, CoinGecko | Daily BTC/USDT close |
| Nasdaq | Yahoo `QQQ` | Nasdaq `QQQ` | Same ETF instrument |
| U.S. dollar | Yahoo `DX-Y.NYB` | Nasdaq `UUP` | UUP is an ETF proxy, not DXY |
| Gold | Yahoo `GC=F` | Nasdaq `GLD` | GLD is an ETF proxy, not COMEX futures |
| Oil | Yahoo `CL=F` | Nasdaq `USO` | USO is an ETF proxy, not WTI futures |
| 10-year real yield | FRED `DFII10` | U.S. Treasury | Business-day publication |

## Instrument Consistency

When Yahoo is unavailable, the application rebuilds the complete macro feature
history using the Nasdaq ETF set. It does not append a few proxy observations to
a futures history. This avoids an artificial return jump at the source boundary.

Switching between futures and ETF proxies changes the economic exposure and the
model coefficients. Record the active source set when comparing experiments.

## Weekend Treatment

Bitcoin trades every day. Traditional-market and macro series do not. The app
forward-fills the latest published observation for weekend scoring and retains
the source's true publication date. A forward-filled value is availability
handling, not a new market observation.

## Cache Files

These local runtime files are intentionally ignored by Git:

- `backup_data.csv`
- `backup_data.meta.json`
- `real_yield_cache.csv`

They should not be treated as authoritative datasets or committed as model
fixtures.
