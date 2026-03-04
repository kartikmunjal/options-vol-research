"""
Options chain and historical price data fetcher.

Data sources:
  - yfinance: free EOD options chains + historical OHLCV
  - FRED: risk-free rate (3-month T-bill / SOFR)

Design philosophy:
  - Cache everything to parquet to avoid redundant API calls
  - Always store raw data before cleaning — reproducibility first
  - Log data quality metrics at fetch time
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# ------------------------------------------------------------------
# Risk-free rate
# ------------------------------------------------------------------

def get_risk_free_rate() -> float:
    """
    Fetch the current 3-month T-bill rate from yfinance (^IRX).
    Returns annualized continuous rate.
    IRX is quoted as % annualized (e.g., 5.25 → 0.0525).
    """
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        rate_pct = float(hist["Close"].dropna().iloc[-1])
        return rate_pct / 100.0
    except Exception as e:
        log.warning(f"Could not fetch T-bill rate: {e}. Defaulting to 5%.")
        return 0.05


# ------------------------------------------------------------------
# Options chain
# ------------------------------------------------------------------

def fetch_options_chain(
    ticker: str,
    as_of: str | None = None,
    cache: bool = True,
) -> tuple[pd.DataFrame, float, float]:
    """
    Fetch the full options chain for a given ticker.

    Parameters
    ----------
    ticker : stock symbol (e.g., 'SPY')
    as_of  : date string 'YYYY-MM-DD' (for cache key only; yfinance returns live data)
    cache  : if True, save raw chain to parquet

    Returns
    -------
    chain  : DataFrame with columns standardized for vol surface construction
    spot   : current spot price
    r      : risk-free rate (annualized)
    """
    today = as_of or date.today().isoformat()
    cache_path = RAW_DIR / f"chain_{ticker}_{today}.parquet"

    if cache and cache_path.exists():
        log.info(f"Loading cached chain from {cache_path}")
        chain = pd.read_parquet(cache_path)
        spot = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        return chain, float(spot), get_risk_free_rate()

    t = yf.Ticker(ticker)

    # Current spot
    hist = t.history(period="2d")
    spot = float(hist["Close"].iloc[-1])
    r = get_risk_free_rate()

    log.info(f"Fetching {ticker} options chain. Spot={spot:.2f}, r={r:.3f}")

    # Fetch all available expiries
    expiry_dates = t.options
    if not expiry_dates:
        raise ValueError(f"No options data available for {ticker}")

    all_rows = []
    today_dt = datetime.strptime(today, "%Y-%m-%d").date()

    for exp_str in expiry_dates:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
        T = (exp_dt - today_dt).days / 365.0

        if T <= 0:
            continue

        try:
            chain_exp = t.option_chain(exp_str)
        except Exception as e:
            log.warning(f"Failed to fetch {ticker} expiry {exp_str}: {e}")
            continue

        for opt_type, df in [("call", chain_exp.calls), ("put", chain_exp.puts)]:
            df = df.copy()
            df["option_type"] = opt_type
            df["expiry"] = exp_str
            df["expiry_years"] = T
            df["log_moneyness"] = np.log(df["strike"] / spot)
            all_rows.append(df)

    if not all_rows:
        raise ValueError(f"No options data fetched for {ticker}")

    chain = pd.concat(all_rows, ignore_index=True)
    chain["fetch_date"] = today
    chain["spot"] = spot

    # Standardize column names
    chain = chain.rename(columns={
        "openInterest": "open_interest",
        "lastTradeDate": "last_trade_date",
        "impliedVolatility": "yf_iv",
        "inTheMoney": "in_the_money",
        "lastPrice": "last_price",
        "percentChange": "pct_change",
    })

    # Ensure numeric types
    for col in ["bid", "ask", "volume", "open_interest", "strike"]:
        if col in chain.columns:
            chain[col] = pd.to_numeric(chain[col], errors="coerce").fillna(0)

    chain["mid_price"] = (chain["bid"] + chain["ask"]) / 2

    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        chain.to_parquet(cache_path, index=False)
        log.info(f"Saved chain ({len(chain)} rows) to {cache_path}")

    _log_chain_stats(chain, ticker, spot)

    return chain, spot, r


# ------------------------------------------------------------------
# Historical prices (for realized vol calculation)
# ------------------------------------------------------------------

def fetch_historical_prices(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV history for realized vol calculations.

    Parameters
    ----------
    period   : yfinance period string ('1y', '2y', 'max')
    interval : '1d', '1h', '5m'

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume, returns, log_returns]
    """
    cache_path = RAW_DIR / f"prices_{ticker}_{period}_{interval}.parquet"

    if cache and cache_path.exists():
        # Refresh if stale (older than 1 day)
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime).days < 1:
            return pd.read_parquet(cache_path)

    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df.columns = [c.lower() for c in df.columns]
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

    return df


# ------------------------------------------------------------------
# Realized volatility computation
# ------------------------------------------------------------------

def realized_vol(
    prices: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """
    Rolling realized volatility using close-to-close log returns.

    The standard estimator: σ_R = std(log returns) × √252

    More sophisticated alternatives (not yet implemented here):
    - Parkinson (1980): uses high-low range → 30% more efficient
    - Garman-Klass (1980): uses OHLC → 7× more efficient
    - Yang-Zhang (2000): handles overnight jumps

    window : rolling window in trading days (21 ≈ 1 month)
    """
    lr = prices["log_returns"].dropna()
    rv = lr.rolling(window).std()
    if annualize:
        rv *= np.sqrt(252)
    return rv


def vol_risk_premium(
    prices: pd.DataFrame,
    chain: pd.DataFrame,
    atm_iv: float,
    window: int = 21,
) -> pd.DataFrame:
    """
    Compute the Variance Risk Premium (VRP): IV - RV.

    VRP > 0 historically: implied vol is ~2-3 vol points above realized vol
    on average for SPY. This is the insurance premium collected by option sellers
    and the core P&L driver of volatility selling strategies.

    Returns a DataFrame tracking: iv, rv, vrp over time.
    """
    rv = realized_vol(prices, window=window)
    rv_df = rv.to_frame("rv")
    rv_df["atm_iv"] = atm_iv  # snapshot — in practice, fetch from surface daily
    rv_df["vrp"] = rv_df["atm_iv"] - rv_df["rv"]
    return rv_df.dropna()


# ------------------------------------------------------------------
# Private
# ------------------------------------------------------------------

def _log_chain_stats(chain: pd.DataFrame, ticker: str, spot: float) -> None:
    n_expiries = chain["expiry"].nunique()
    n_calls = (chain["option_type"] == "call").sum()
    n_puts = (chain["option_type"] == "put").sum()
    min_T = chain["expiry_years"].min()
    max_T = chain["expiry_years"].max()

    log.info(
        f"{ticker} chain: {len(chain)} options | "
        f"{n_calls} calls, {n_puts} puts | "
        f"{n_expiries} expiries ({min_T*365:.0f}d - {max_T*365:.0f}d) | "
        f"spot={spot:.2f}"
    )
