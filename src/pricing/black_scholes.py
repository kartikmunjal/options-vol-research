"""
Black-Scholes pricing model with analytic Greeks.

All formulas derived from first principles. No black-box library calls —
every Greek is computed from the closed-form solution to the BS PDE.

Notation:
    S  : spot price
    K  : strike price
    T  : time to expiry (in years)
    r  : risk-free rate (continuous compounding)
    q  : continuous dividend yield
    σ  : implied / realized volatility (annualized)
    φ  : standard normal PDF
    Φ  : standard normal CDF
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Standardized moneyness — drives all Greek expressions."""
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * np.sqrt(T)


# ------------------------------------------------------------------
# Price
# ------------------------------------------------------------------

def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """
    Black-Scholes-Merton option price.

    Parameters
    ----------
    S, K, T, r, sigma : standard BSM inputs
    option_type       : 'call' or 'put'
    q                 : continuous dividend yield (default 0)

    Returns
    -------
    float : fair value of the option
    """
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return float(intrinsic)

    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ------------------------------------------------------------------
# First-order Greeks
# ------------------------------------------------------------------

def delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0
) -> float:
    """
    dV/dS — sensitivity to spot move.

    Call delta ∈ (0, 1); put delta ∈ (-1, 0).
    At-the-money call delta ≈ 0.5 (slightly above due to σ√T convexity).
    """
    if T <= 0:
        return 1.0 if (option_type == "call" and S > K) else 0.0

    d1 = _d1(S, K, T, r, q, sigma)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1)


def vega(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """
    dV/dσ — sensitivity to a 1-unit (100 vol point) move in vol.
    Same for calls and puts by put-call parity.

    Divide by 100 to get sensitivity per 1 vol point (i.e., 1%).
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0
) -> float:
    """
    dV/dt — daily time decay (per calendar day).

    Theta is negative for long options — the option loses value as time passes.
    Theta and gamma have a precise relationship: Θ + ½σ²S²Γ = rV (BS PDE).
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)

    common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        annual = (
            common
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
    else:
        annual = (
            common
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )

    return annual / 365.0  # per calendar day


# ------------------------------------------------------------------
# Second-order Greeks
# ------------------------------------------------------------------

def gamma(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """
    d²V/dS² — rate of change of delta with spot.

    Same for calls and puts. Gamma is always positive for long options.
    Key intuition: gamma is your friend when you OWN options (long gamma),
    your enemy when you've SOLD them (short gamma). Gamma P&L ≈ ½Γ(ΔS)².
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vanna(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """
    d²V/(dS dσ) = dDelta/dσ = dVega/dS.

    Vanna matters for hedging when vol and spot move together (skew P&L).
    Negative vanna on OTM puts is why skew exists — dealers hedge vanna risk
    by buying more puts, driving up their implied vol.
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


def volga(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """
    d²V/dσ² = dVega/dσ — vol convexity (also called vomma).

    Positive volga means you benefit from vol-of-vol.
    OTM options have higher volga than ATM → vol surface smile partly explained
    by volga risk premium demanded by dealers.
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)
    return vega(S, K, T, r, sigma, q) * d1 * d2 / sigma


def charm(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0
) -> float:
    """
    dDelta/dt — rate of change of delta over time (delta bleed).

    Critical for daily delta-hedging: your hedge ratio drifts even if spot
    doesn't move. ATM charm ≈ 0; OTM/ITM charm can be large near expiry.
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)

    shared = np.exp(-q * T) * norm.pdf(d1) * (
        2 * (r - q) * T - d2 * sigma * np.sqrt(T)
    ) / (2 * T * sigma * np.sqrt(T))

    if option_type == "call":
        return q * np.exp(-q * T) * norm.cdf(d1) - shared
    else:
        return -q * np.exp(-q * T) * norm.cdf(-d1) - shared


# ------------------------------------------------------------------
# Convenience: full Greeks bundle
# ------------------------------------------------------------------

def all_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0
) -> dict[str, float]:
    """Return all Greeks in a single dict."""
    return {
        "price":  price(S, K, T, r, sigma, option_type, q),
        "delta":  delta(S, K, T, r, sigma, option_type, q),
        "gamma":  gamma(S, K, T, r, sigma, q),
        "theta":  theta(S, K, T, r, sigma, option_type, q),
        "vega":   vega(S, K, T, r, sigma, q),
        "vanna":  vanna(S, K, T, r, sigma, q),
        "volga":  volga(S, K, T, r, sigma, q),
        "charm":  charm(S, K, T, r, sigma, option_type, q),
    }
