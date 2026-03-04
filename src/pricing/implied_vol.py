"""
Implied volatility solver.

Uses Newton-Raphson with Vega as the Jacobian for fast convergence near ATM,
with a bracket + bisection fallback for deep ITM/OTM options where NR can
overshoot into negative volatility territory.

Key insight: implied vol is the market's consensus forecast of future realized
vol, baked into the option price. IV > RV means you're overpaying for
insurance; IV < RV means the market is under-pricing risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pricing.black_scholes import price as bs_price, vega as bs_vega


# ------------------------------------------------------------------
# Single-option solver
# ------------------------------------------------------------------

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Solve for implied volatility given an observed market price.

    Algorithm
    ---------
    1. Intrinsic value check — price must exceed intrinsic to have finite IV.
    2. Initial guess via Brenner-Subrahmanyam approximation:
           σ₀ ≈ √(2π/T) · (C/S)   (accurate to ~5% for ATM options)
    3. Newton-Raphson: σₙ₊₁ = σₙ - (BS(σₙ) - price) / Vega(σₙ)
    4. Fallback to bisection if NR diverges.

    Parameters
    ----------
    market_price : observed mid-price of the option
    tol          : convergence tolerance in vol units (default 1e-6 ≈ 0.0001%)
    max_iter     : maximum iterations before returning NaN

    Returns
    -------
    float : implied vol (annualized), or NaN if unsolvable
    """
    if T <= 0:
        return float("nan")

    # --- intrinsic check ---
    intrinsic = (
        max(S - K * np.exp(-r * T), 0) if option_type == "call"
        else max(K * np.exp(-r * T) - S, 0)
    )
    if market_price <= intrinsic:
        return float("nan")

    # --- Brenner-Subrahmanyam initial guess ---
    sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
    sigma = np.clip(sigma, 1e-4, 5.0)

    # --- Newton-Raphson ---
    for _ in range(max_iter):
        p = bs_price(S, K, T, r, sigma, option_type, q)
        v = bs_vega(S, K, T, r, sigma, q)

        diff = p - market_price
        if abs(diff) < tol:
            return sigma

        if abs(v) < 1e-10:
            break  # vega ≈ 0 → NR unusable, fall through to bisection

        sigma -= diff / v
        if sigma <= 0:
            break  # overshot, fall through to bisection

    # --- Bisection fallback ---
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = bs_price(S, K, T, r, mid, option_type, q)
        if abs(p - market_price) < tol:
            return mid
        if p < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return 0.5 * (lo + hi)

    return float("nan")


# ------------------------------------------------------------------
# Vectorized surface solver
# ------------------------------------------------------------------

def implied_vol_surface(
    chain: pd.DataFrame,
    S: float,
    r: float = 0.05,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    Compute implied vols for an entire options chain.

    Parameters
    ----------
    chain : DataFrame with columns [strike, expiry_years, mid_price, option_type]
    S     : current spot price
    r     : risk-free rate
    q     : dividend yield

    Returns
    -------
    DataFrame with added column `iv` (implied vol, annualized)
    """
    chain = chain.copy()

    ivs = []
    for _, row in chain.iterrows():
        iv = implied_vol(
            market_price=row["mid_price"],
            S=S,
            K=row["strike"],
            T=row["expiry_years"],
            r=r,
            option_type=row["option_type"],
            q=q,
        )
        ivs.append(iv)

    chain["iv"] = ivs
    return chain.dropna(subset=["iv"])


# ------------------------------------------------------------------
# Sanity check utility
# ------------------------------------------------------------------

def round_trip_error(
    iv: float, market_price: float, S: float, K: float, T: float,
    r: float, option_type: str = "call", q: float = 0.0
) -> float:
    """
    Given a computed IV, reprice and return |BS(IV) - market_price|.
    Should be < 1e-6 for a correct solve.
    """
    return abs(bs_price(S, K, T, r, iv, option_type, q) - market_price)
