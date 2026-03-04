"""
Vol surface construction from raw options chain data.

Pipeline:
  1. Clean chain (filter by liquidity, remove crossed markets)
  2. Compute mid-prices and bid-ask spreads
  3. Solve implied vols for each option
  4. Detect and remove arbitrage violations
  5. Fit SVI parametrization per expiry slice
  6. Expose surface as a callable (strike, T) → IV
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pricing.implied_vol import implied_vol_surface
from src.vol_surface.svi import calibrate_surface, surface_iv


class VolSurface:
    """
    Represents a complete implied volatility surface.

    Usage
    -----
    vs = VolSurface.from_chain(chain, spot=450.0, r=0.05)
    iv = vs.iv(strike=450, T=30/365)          # single point
    smile = vs.smile(T=30/365)                # full smile at expiry
    term_struct = vs.atm_term_structure()     # ATM vol by expiry
    """

    def __init__(
        self,
        chain: pd.DataFrame,
        calibration: dict,
        spot: float,
        r: float,
        q: float,
        build_date: str,
    ):
        self.chain = chain
        self.calibration = calibration
        self.spot = spot
        self.r = r
        self.q = q
        self.build_date = build_date
        self.expiries = sorted(calibration.keys())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_chain(
        cls,
        raw_chain: pd.DataFrame,
        spot: float,
        r: float = 0.05,
        q: float = 0.013,  # SPY dividend yield ≈ 1.3%
        build_date: str = "",
        min_open_interest: int = 100,
        min_volume: int = 10,
        max_spread_pct: float = 0.25,
    ) -> "VolSurface":
        """
        Build a complete vol surface from a raw options chain DataFrame.

        Filtering criteria (based on market microstructure research):
        - Open interest ≥ 100: ensures secondary market liquidity
        - Volume ≥ 10: ensures the quote was traded today
        - Spread / mid ≤ 25%: removes stale/wide market-maker quotes
        - T ≥ 3 days: removes same-week expiries (gamma blows up)
        - Intrinsic violation: removes options priced below intrinsic value
        """
        chain = cls._clean_chain(
            raw_chain, spot, r, q,
            min_open_interest, min_volume, max_spread_pct
        )

        # Compute implied vols
        chain = implied_vol_surface(chain, spot, r=r, q=q)

        # Filter bad IVs
        chain = chain[(chain["iv"] > 0.02) & (chain["iv"] < 2.5)]

        # Fit SVI per expiry
        calibration = calibrate_surface(chain, spot, r=r, q=q)

        return cls(chain, calibration, spot, r, q, build_date)

    # ------------------------------------------------------------------
    # Querying the surface
    # ------------------------------------------------------------------

    def iv(self, strike: float, T: float) -> float:
        """Interpolate implied vol at (strike, T)."""
        return float(
            surface_iv(
                np.array([strike]), T, self.calibration,
                self.spot, self.r, self.q
            )[0]
        )

    def smile(
        self,
        T: float,
        moneyness_range: tuple[float, float] = (-0.3, 0.3),
        n_points: int = 100,
    ) -> pd.DataFrame:
        """
        Return the volatility smile at a given expiry.

        Parameters
        ----------
        T              : expiry in years (uses nearest calibrated slice)
        moneyness_range: log-moneyness range (default: ±30%)
        n_points       : resolution

        Returns
        -------
        DataFrame with columns: [strike, log_moneyness, iv, delta]
        """
        # Nearest calibrated expiry
        T_cal = min(self.expiries, key=lambda t: abs(t - T))
        F = self.calibration[T_cal]["forward"]

        k_grid = np.linspace(*moneyness_range, n_points)
        strikes = F * np.exp(k_grid)
        ivs = surface_iv(strikes, T, self.calibration, self.spot, self.r, self.q)

        from src.pricing.black_scholes import delta as bs_delta
        deltas = [bs_delta(self.spot, K, T_cal, self.r, iv, "call", self.q)
                  for K, iv in zip(strikes, ivs)]

        return pd.DataFrame({
            "strike": strikes,
            "log_moneyness": k_grid,
            "iv": ivs,
            "delta": deltas,
        })

    def atm_term_structure(self) -> pd.DataFrame:
        """
        Return ATM implied vol at each calibrated expiry.

        Term structure encodes:
        - Near-term: realized vol expectations (earnings, FOMC)
        - Long-term: long-run vol estimates (structural level)
        - Inversion: front-month > back-month → stress/event risk
        """
        rows = []
        for T in self.expiries:
            result = self.calibration[T]
            F = result["forward"]
            iv_atm = self.iv(F, T)  # IV at the forward (ATM)
            rows.append({
                "expiry_years": T,
                "expiry_days": round(T * 365),
                "forward": F,
                "atm_iv": iv_atm,
                "svi_rmse": result.get("rmse", np.nan),
                "arb_free": result.get("arbitrage_check", {}).get("arbitrage_free", None),
            })
        return pd.DataFrame(rows)

    def skew(self, T: float, delta_otm: float = 0.25) -> dict:
        """
        Compute the 25-delta skew and risk reversal.

        Risk reversal = IV(25Δ put) - IV(25Δ call)
        Positive RR → put skew (typical in equities — crash risk premium)

        Butterfly = [IV(25Δ put) + IV(25Δ call)] / 2 - IV(50Δ)
        Positive butterfly → smile convexity (vol of vol premium)
        """
        smile_df = self.smile(T)

        atm_iv = smile_df.iloc[(smile_df["delta"] - 0.50).abs().argsort()[:1]]["iv"].values[0]
        put25_iv = smile_df.iloc[(smile_df["delta"] - (1 - delta_otm)).abs().argsort()[:1]]["iv"].values[0]
        call25_iv = smile_df.iloc[(smile_df["delta"] - delta_otm).abs().argsort()[:1]]["iv"].values[0]

        return {
            "atm_iv": atm_iv,
            "put25_iv": put25_iv,
            "call25_iv": call25_iv,
            "risk_reversal": put25_iv - call25_iv,
            "butterfly": 0.5 * (put25_iv + call25_iv) - atm_iv,
        }

    def surface_grid(
        self,
        k_range: tuple[float, float] = (-0.4, 0.4),
        n_k: int = 60,
        n_T: int = None,
    ) -> pd.DataFrame:
        """Return IV on a regular (k, T) grid — useful for 3D plotting."""
        k_grid = np.linspace(*k_range, n_k)
        T_grid = self.expiries if n_T is None else np.linspace(min(self.expiries), max(self.expiries), n_T)

        rows = []
        for T in T_grid:
            T_cal = min(self.expiries, key=lambda t: abs(t - T))
            F = self.calibration[T_cal]["forward"]
            strikes = F * np.exp(k_grid)
            ivs = surface_iv(strikes, T, self.calibration, self.spot, self.r, self.q)
            for k, iv in zip(k_grid, ivs):
                rows.append({"T": T, "log_moneyness": k, "iv": iv})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_chain(
        raw: pd.DataFrame,
        S: float,
        r: float,
        q: float,
        min_oi: int,
        min_vol: int,
        max_spread_pct: float,
    ) -> pd.DataFrame:
        """Apply market microstructure filters to raw options chain."""
        df = raw.copy()

        # Standardize column names (yfinance format)
        df = df.rename(columns={
            "openInterest": "open_interest",
            "lastTradeDate": "last_trade",
            "impliedVolatility": "yf_iv",
            "inTheMoney": "itm",
        })

        # Mid price + spread
        df["mid_price"] = (df["bid"] + df["ask"]) / 2
        df["spread"] = df["ask"] - df["bid"]
        df["spread_pct"] = df["spread"] / df["mid_price"].clip(lower=0.01)

        # Liquidity filters
        mask = (
            (df["open_interest"] >= min_oi)
            & (df["volume"] >= min_vol)
            & (df["spread_pct"] <= max_spread_pct)
            & (df["bid"] > 0)
            & (df["mid_price"] > 0.05)  # > 5 cents minimum
        )
        df = df[mask].copy()

        # Time filter: at least 3 days to expiry
        df = df[df["expiry_years"] >= 3 / 365]

        # Remove extreme moneyness (too deep ITM/OTM → unreliable IV)
        df = df[df["log_moneyness"].abs() <= 0.6]

        return df.reset_index(drop=True)
