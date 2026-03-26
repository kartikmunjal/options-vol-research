"""
Delta-hedging backtest with full P&L attribution.

Strategy: Sell 1 ATM straddle on SPY, delta-hedge daily.

P&L Decomposition (from the BS PDE):
    dV ≈ Δ·dS + ½Γ(dS)² + Θ·dt + ν·dσ

For a SHORT straddle:
    P&L = -[Δ·dS + ½Γ(dS)² + Θ·dt + ν·dσ] + hedge_PnL
         = -Gamma P&L - Theta P&L - Vega P&L + transaction costs

At expiry, the net P&L from delta-hedging an option is approximately:
    P&L_total ≈ ½ · Γ · S² · (RV² - IV²) · dt  [summed over all rebalance periods]

This is the CORE INSIGHT: you profit when RV < IV (you sold expensive insurance).

The variance risk premium (IV - RV ≈ 2-3 vol points on average for SPY)
is the structural source of alpha for volatility sellers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.pricing._pricer import all_greeks, price as bs_price, delta as bs_delta

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Straddle position
# ------------------------------------------------------------------

@dataclass
class StraddlePosition:
    """A short ATM straddle (sell 1 call + 1 put, same strike, same expiry)."""

    strike: float
    expiry_years: float
    entry_iv: float
    entry_date: str
    entry_spot: float
    contracts: int = 1  # number of contracts (each = 100 shares)
    multiplier: int = 100

    # Entry prices
    entry_call_price: float = 0.0
    entry_put_price: float = 0.0
    entry_total_premium: float = 0.0

    # Hedge
    hedge_shares: float = 0.0  # shares of SPY held to hedge delta
    hedge_cost_basis: float = 0.0

    # Cumulative P&L components
    pnl_delta: float = 0.0
    pnl_gamma: float = 0.0
    pnl_theta: float = 0.0
    pnl_vega: float = 0.0
    pnl_transaction: float = 0.0

    history: list[dict] = field(default_factory=list)

    @property
    def notional(self) -> float:
        return self.contracts * self.multiplier * self.entry_spot


# ------------------------------------------------------------------
# Backtest engine
# ------------------------------------------------------------------

class DeltaHedgeBacktest:
    """
    Walk-forward delta-hedging simulation over historical price data.

    For each simulation day:
      1. Reprice the straddle at current spot + vol (use realized vol estimate)
      2. Compute delta of the total straddle position
      3. Rebalance hedge shares to delta-neutralize
      4. Attribute today's P&L across greeks
      5. Record daily snapshot

    Parameters
    ----------
    prices       : DataFrame with [close, log_returns, ...] indexed by date
    r            : risk-free rate (annualized)
    q            : dividend yield
    hedge_cost   : transaction cost per share in $ (default $0.01)
    rebal_freq   : rebalancing frequency in days (1=daily, 5=weekly)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        r: float = 0.05,
        q: float = 0.013,
        hedge_cost: float = 0.01,
        rebal_freq: int = 1,
    ):
        self.prices = prices.copy()
        self.r = r
        self.q = q
        self.hedge_cost = hedge_cost
        self.rebal_freq = rebal_freq

    def run(
        self,
        entry_date: str,
        days_to_expiry: int = 30,
        initial_iv: float | None = None,
        vol_model: str = "rolling_21d",
    ) -> pd.DataFrame:
        """
        Run a single straddle trade from entry_date to expiry.

        Parameters
        ----------
        entry_date     : simulation start date (YYYY-MM-DD)
        days_to_expiry : DTE at entry (typically 21-45 for theta-decay strategies)
        initial_iv     : IV at entry; if None, use historical vol as proxy
        vol_model      : how to estimate daily vol for repricing
                         'rolling_21d'  : 21-day realized vol
                         'ewm_20d'      : exponentially weighted (faster reacting)
                         'flat'         : keep IV fixed at entry (unrealistic but clean)

        Returns
        -------
        DataFrame with daily P&L attribution
        """
        prices = self.prices[self.prices.index >= entry_date].copy()
        if len(prices) < 2:
            raise ValueError(f"Not enough price data from {entry_date}")

        T0 = days_to_expiry / 365.0
        entry_spot = float(prices["close"].iloc[0])

        # Entry IV
        if initial_iv is None:
            rv = prices["log_returns"].rolling(21).std().iloc[0] * np.sqrt(252)
            initial_iv = float(rv) if not np.isnan(rv) else 0.20
            # Add a vol risk premium estimate (~2 vol points) — IV > RV empirically
            initial_iv += 0.02

        # ATM strike: round to nearest $1 for SPY
        strike = round(entry_spot)

        pos = self._enter_trade(entry_date, entry_spot, strike, T0, initial_iv)
        log.info(
            f"Entered short straddle: K={strike}, T={days_to_expiry}d, "
            f"IV={initial_iv:.1%}, premium=${pos.entry_total_premium:.2f}, "
            f"spot={entry_spot:.2f}"
        )

        # Day-by-day simulation
        records = []
        prev_spot = entry_spot
        prev_iv = initial_iv

        for i, (dt, row) in enumerate(prices.iloc[1:].iterrows(), start=1):
            T_remaining = max(T0 - i / 365.0, 1 / 365.0)
            curr_spot = float(row["close"])

            # Estimate current implied vol
            curr_iv = self._estimate_vol(prices, i, vol_model, initial_iv)

            # Price straddle at current params
            call_px = bs_price(curr_spot, strike, T_remaining, self.r, curr_iv, "call", self.q)
            put_px = bs_price(curr_spot, strike, T_remaining, self.r, curr_iv, "put", self.q)
            straddle_px = call_px + put_px

            # Greeks of the total straddle (call + put, scaled by contracts × multiplier)
            scale = pos.contracts * pos.multiplier
            call_g = all_greeks(curr_spot, strike, T_remaining, self.r, curr_iv, "call", self.q)
            put_g = all_greeks(curr_spot, strike, T_remaining, self.r, curr_iv, "put", self.q)

            total_delta = (call_g["delta"] + put_g["delta"]) * scale
            total_gamma = (call_g["gamma"] + put_g["gamma"]) * scale
            total_theta = (call_g["theta"] + put_g["theta"]) * scale
            total_vega = (call_g["vega"] + put_g["vega"]) * scale

            dS = curr_spot - prev_spot
            dSigma = curr_iv - prev_iv

            # P&L attribution (SHORT position → flip signs)
            pnl_delta_component = -total_delta * dS  # hedged out, but track it
            pnl_gamma_component = -0.5 * total_gamma * dS**2  # SHORT gamma → negative when market moves
            pnl_theta_component = -total_theta  # SHORT → collect theta decay daily
            pnl_vega_component = -total_vega * dSigma  # SHORT vega → hurt by vol expansion

            # Hedge P&L: long hedge_shares of SPY
            hedge_pnl = pos.hedge_shares * dS

            # Total daily P&L on position
            pnl_position = (
                pnl_gamma_component
                + pnl_theta_component
                + pnl_vega_component
            )
            pnl_total = pnl_position + hedge_pnl

            # Rebalance hedge if it's a rebalancing day
            transaction_cost = 0.0
            if i % self.rebal_freq == 0:
                new_hedge = -total_delta  # delta-neutral requires offsetting position delta
                shares_traded = new_hedge - pos.hedge_shares
                transaction_cost = abs(shares_traded) * self.hedge_cost
                pos.hedge_shares = new_hedge

            pnl_total -= transaction_cost

            record = {
                "date": dt,
                "day": i,
                "spot": curr_spot,
                "strike": strike,
                "T_remaining_years": T_remaining,
                "T_remaining_days": T_remaining * 365,
                "iv": curr_iv,
                "straddle_price": straddle_px * scale,
                "delta": total_delta,
                "gamma": total_gamma,
                "theta": total_theta,
                "vega": total_vega,
                "hedge_shares": pos.hedge_shares,
                "dS": dS,
                "dSigma": dSigma,
                "pnl_gamma": pnl_gamma_component,
                "pnl_theta": pnl_theta_component,
                "pnl_vega": pnl_vega_component,
                "pnl_hedge": hedge_pnl,
                "pnl_transaction": -transaction_cost,
                "pnl_total": pnl_total,
            }
            records.append(record)

            # Check if expired
            if T_remaining <= 1.5 / 365:
                log.info(f"Position expired on {dt}. Final spot={curr_spot:.2f}, K={strike}")
                break

            prev_spot = curr_spot
            prev_iv = curr_iv

        results = pd.DataFrame(records)
        results["cumulative_pnl"] = results["pnl_total"].cumsum()
        results["cumulative_gamma"] = results["pnl_gamma"].cumsum()
        results["cumulative_theta"] = results["pnl_theta"].cumsum()
        results["cumulative_vega"] = results["pnl_vega"].cumsum()
        results["cumulative_transaction"] = results["pnl_transaction"].cumsum()

        return results

    def run_rolling(
        self,
        start_date: str,
        end_date: str,
        days_to_expiry: int = 30,
        roll_days_before_expiry: int = 5,
    ) -> pd.DataFrame:
        """
        Run a continuous rolling short straddle strategy.

        Roll (close + re-enter) `roll_days_before_expiry` days before expiry,
        to avoid gamma risk into expiration.
        """
        all_results = []
        entry = start_date
        trade_id = 0

        while entry < end_date:
            try:
                result = self.run(entry, days_to_expiry)
                result["trade_id"] = trade_id
                all_results.append(result)

                # Next entry: DTE - roll_days_before_expiry after current entry
                expiry_day = len(result)
                if expiry_day < (days_to_expiry - roll_days_before_expiry):
                    break

                last_date = result["date"].iloc[-1]
                entry = pd.Timestamp(last_date).strftime("%Y-%m-%d")
                trade_id += 1

            except Exception as e:
                log.warning(f"Trade starting {entry} failed: {e}")
                break

        if not all_results:
            raise ValueError("No successful trades in the backtest period.")

        combined = pd.concat(all_results, ignore_index=True)
        combined["strategy_cumulative_pnl"] = combined["pnl_total"].cumsum()
        return combined

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(results: pd.DataFrame, initial_notional: float) -> dict:
        """
        Compute key performance metrics from backtest results.

        Metrics:
        - Total P&L and return on notional
        - Sharpe ratio (annualized)
        - P&L attribution breakdown (theta vs gamma vs vega)
        - Win rate and average win/loss
        - Max drawdown
        """
        pnl = results["pnl_total"]
        cum_pnl = results["cumulative_pnl"]
        n_days = len(results)

        # Returns
        daily_ret = pnl / initial_notional
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

        # Drawdown
        cum_max = cum_pnl.cummax()
        drawdown = cum_pnl - cum_max
        max_dd = drawdown.min()
        max_dd_pct = max_dd / initial_notional

        # Attribution
        total_pnl = pnl.sum()
        attribution = {
            "theta": results["pnl_theta"].sum(),
            "gamma": results["pnl_gamma"].sum(),
            "vega": results["pnl_vega"].sum(),
            "transaction": results["pnl_transaction"].sum(),
            "hedge": results["pnl_hedge"].sum(),
        }

        # Win rate
        win_rate = (pnl > 0).mean()

        return {
            "total_pnl": total_pnl,
            "return_on_notional": total_pnl / initial_notional,
            "annualized_return": (total_pnl / initial_notional) * (252 / n_days),
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd_pct,
            "win_rate": win_rate,
            "n_days": n_days,
            "pnl_attribution": attribution,
            "pnl_attribution_pct": {
                k: v / abs(total_pnl) if abs(total_pnl) > 0 else 0
                for k, v in attribution.items()
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enter_trade(
        self, date: str, spot: float, strike: float, T: float, iv: float
    ) -> StraddlePosition:
        pos = StraddlePosition(
            strike=strike, expiry_years=T, entry_iv=iv,
            entry_date=date, entry_spot=spot,
        )
        scale = pos.contracts * pos.multiplier
        call_px = bs_price(spot, strike, T, self.r, iv, "call", self.q)
        put_px = bs_price(spot, strike, T, self.r, iv, "put", self.q)
        pos.entry_call_price = call_px
        pos.entry_put_price = put_px
        pos.entry_total_premium = (call_px + put_px) * scale

        # Initial delta: straddle delta ≈ 0 (ATM call delta + put delta ≈ 0.5 - 0.5)
        call_d = bs_delta(spot, strike, T, self.r, iv, "call", self.q)
        put_d = bs_delta(spot, strike, T, self.r, iv, "put", self.q)
        straddle_delta = (call_d + put_d) * scale
        pos.hedge_shares = -straddle_delta  # short position, hedge by going long spot

        return pos

    def _estimate_vol(
        self, prices: pd.DataFrame, current_idx: int, model: str, initial_iv: float
    ) -> float:
        """Estimate current implied vol for repricing."""
        if model == "flat":
            return initial_iv

        lr = prices["log_returns"].iloc[:current_idx + 1]

        if model == "rolling_21d":
            window = min(21, len(lr))
            rv = lr.rolling(window).std().iloc[-1] * np.sqrt(252)
        elif model == "ewm_20d":
            rv = lr.ewm(span=20).std().iloc[-1] * np.sqrt(252)
        else:
            rv = initial_iv

        if np.isnan(rv) or rv <= 0:
            return initial_iv

        # Add estimated vol risk premium: IV ≈ RV + VRP
        vrp = 0.02  # 2 vol points historical average for SPY
        return float(rv + vrp)
