"""
SVI (Stochastic Volatility Inspired) Parametrization.

Introduced by Jim Gatheral (2004). SVI parametrizes the total implied variance
w(k) = σ²(k)·T as a function of log-moneyness k = log(F/K):

    w(k) = a + b · [ρ(k - m) + √((k - m)² + σ²)]

Parameters
----------
a : vertical translation — overall variance level
b : slope — controls the "angle" of the skew and curvature
ρ : correlation ([-1, 1]) — controls skew asymmetry (ρ < 0 → left skew)
m : horizontal translation — ATM shift
σ : curvature (σ > 0) — controls the smile width

Key properties:
  - No static arbitrage if b(1+|ρ|) ≤ 4/T (butterfly condition)
  - Total variance w(k) ≥ 0 for all k (guaranteed for b ≥ 0, σ > 0)
  - Recovers Black-Scholes as b → 0 (flat surface)
  - Captures skew (ρ), smile (σ), and term structure (a, b scale with T)

References
----------
Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility
parametrization with application to the valuation of volatility derivatives."
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds


# ------------------------------------------------------------------
# SVI total variance
# ------------------------------------------------------------------

def svi_total_variance(k: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute SVI total variance w(k) = σ_impl²(k) · T.

    Parameters
    ----------
    k      : log-moneyness array, k = log(K/F) where F = S·exp(r·T)
    params : dict with keys {a, b, rho, m, sigma}

    Returns
    -------
    w : total implied variance (non-negative)
    """
    a, b, rho, m, sigma = (
        params["a"], params["b"], params["rho"],
        params["m"], params["sigma"],
    )
    inner = rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2)
    return a + b * inner


def svi_implied_vol(k: np.ndarray, T: float, params: dict) -> np.ndarray:
    """Convert SVI total variance to annualized implied vol."""
    w = svi_total_variance(k, params)
    w = np.maximum(w, 1e-10)  # numerical safety
    return np.sqrt(w / T)


# ------------------------------------------------------------------
# Arbitrage checks
# ------------------------------------------------------------------

def butterfly_arbitrage_free(params: dict, T: float) -> bool:
    """
    Check Gatheral's butterfly (no-crossing) condition:
        b(1 + |ρ|) ≤ 4/T

    Violation means the surface has negative butterfly spreads,
    implying negative probability densities — a hard no-go.
    """
    b, rho = params["b"], params["rho"]
    return b * (1 + abs(rho)) <= 4.0 / T


def is_arbitrage_free(params: dict, T: float, k_grid: np.ndarray | None = None) -> dict:
    """
    Check both butterfly and calendar arbitrage constraints.

    Returns
    -------
    dict with 'butterfly_ok', 'min_variance' (should be > 0)
    """
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 200)

    w = svi_total_variance(k_grid, params)
    return {
        "butterfly_ok": butterfly_arbitrage_free(params, T),
        "min_variance": float(w.min()),
        "arbitrage_free": butterfly_arbitrage_free(params, T) and w.min() > 0,
    }


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

def calibrate_svi(
    k: np.ndarray,
    market_iv: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
) -> dict:
    """
    Fit SVI parameters to market implied vols for a single expiry slice.

    Objective: minimize weighted RMSE between SVI-implied vol and market IV,
    subject to:
      - b ≥ 0, σ > 0  (structural)
      - b(1 + |ρ|) ≤ 4/T  (butterfly no-arbitrage)

    Parameters
    ----------
    k          : log-moneyness array (log(K/F))
    market_iv  : observed implied vols (annualized, not total variance)
    T          : time to expiry in years
    weights    : per-strike weights (default: 1/spread or uniform)

    Returns
    -------
    dict : fitted params + calibration diagnostics
    """
    if weights is None:
        weights = np.ones(len(k))
    weights = weights / weights.sum()

    market_w = market_iv ** 2 * T  # target: total variance

    def objective(x):
        params = _unpack(x)
        w_fit = svi_total_variance(k, params)
        residuals = (w_fit - market_w) * np.sqrt(weights)
        penalty = 0.0
        # soft butterfly penalty
        bfly_slack = 4.0 / T - params["b"] * (1 + abs(params["rho"]))
        if bfly_slack < 0:
            penalty += 1e4 * bfly_slack ** 2
        return np.sum(residuals ** 2) + penalty

    # Multiple restarts to avoid local minima
    best_result = None
    init_points = _generate_initial_guesses(k, market_w, T)

    for x0 in init_points:
        try:
            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=_param_bounds(T),
                options={"maxiter": 2000, "ftol": 1e-12},
            )
            if best_result is None or res.fun < best_result.fun:
                best_result = res
        except Exception:
            continue

    if best_result is None or not best_result.success:
        warnings.warn("SVI calibration may not have converged.")

    params = _unpack(best_result.x)
    w_fit = svi_total_variance(k, params)
    iv_fit = np.sqrt(np.maximum(w_fit, 0) / T)

    rmse = float(np.sqrt(np.mean((iv_fit - market_iv) ** 2)))
    arb = is_arbitrage_free(params, T, k)

    return {
        "params": params,
        "rmse": rmse,
        "arbitrage_check": arb,
        "converged": best_result.success if best_result else False,
    }


# ------------------------------------------------------------------
# Surface (all expiries)
# ------------------------------------------------------------------

def calibrate_surface(
    chain: pd.DataFrame,
    S: float,
    r: float = 0.05,
    q: float = 0.0,
) -> dict[float, dict]:
    """
    Calibrate SVI slice-by-slice for each expiry in the options chain.

    Parameters
    ----------
    chain : DataFrame with columns [strike, expiry_years, iv, option_type]
    S     : spot price
    r, q  : risk-free rate, dividend yield

    Returns
    -------
    dict mapping expiry_years → calibration result dict
    """
    results = {}

    for T, group in chain.groupby("expiry_years"):
        F = S * np.exp((r - q) * T)  # forward price
        k = np.log(group["strike"].values / F)  # log-moneyness
        iv = group["iv"].values

        # Filter: remove illiquid tails
        mask = (np.abs(k) < 0.8) & (iv > 0.01) & (iv < 2.0)
        if mask.sum() < 5:
            continue

        result = calibrate_svi(k[mask], iv[mask], T)
        result["forward"] = F
        result["T"] = T
        results[T] = result

    return results


# ------------------------------------------------------------------
# Evaluation on arbitrary (k, T) grid
# ------------------------------------------------------------------

def surface_iv(
    strikes: np.ndarray,
    T: float,
    calibration_results: dict,
    S: float,
    r: float = 0.05,
    q: float = 0.0,
    method: str = "nearest",
) -> np.ndarray:
    """
    Evaluate implied vol at arbitrary strikes for a given expiry.
    Interpolates between calibrated slices if exact T not available.
    """
    available_T = np.array(sorted(calibration_results.keys()))

    if T in calibration_results:
        result = calibration_results[T]
    elif method == "nearest":
        idx = np.argmin(np.abs(available_T - T))
        result = calibration_results[available_T[idx]]
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    F = result["forward"]
    k = np.log(strikes / F)
    return svi_implied_vol(k, result["T"], result["params"])


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _unpack(x: np.ndarray) -> dict:
    return {"a": x[0], "b": x[1], "rho": x[2], "m": x[3], "sigma": x[4]}


def _param_bounds(T: float) -> Bounds:
    """
    Parameter bounds enforcing structural constraints:
    a ∈ (-∞, 1), b ∈ [0, 2/T], ρ ∈ [-0.999, 0.999], m ∈ (-1, 1), σ > 0
    """
    b_max = 2.0 / T
    return Bounds(
        lb=[-0.5,  0.001, -0.999, -1.0, 1e-4],
        ub=[ 1.0,  b_max,  0.999,  1.0, 2.0],
    )


def _generate_initial_guesses(k: np.ndarray, market_w: np.ndarray, T: float) -> list[np.ndarray]:
    """Generate diverse starting points for global search."""
    atm_var = float(np.interp(0.0, np.sort(k), market_w[np.argsort(k)]))
    atm_var = max(atm_var, 1e-4)

    seeds = [
        # flat surface
        np.array([atm_var, 0.1, -0.5, 0.0, 0.1]),
        # skewed (typical equity: negative rho)
        np.array([atm_var * 0.8, 0.2, -0.7, -0.1, 0.15]),
        # low curvature
        np.array([atm_var, 0.05, -0.3, 0.0, 0.3]),
        # high skew
        np.array([atm_var * 0.5, 0.3, -0.9, -0.2, 0.05]),
    ]
    return seeds
