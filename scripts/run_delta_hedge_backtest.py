#!/usr/bin/env python3
"""
Delta-hedging backtest: short ATM straddle on SPY, daily delta rebalancing.

Core hypothesis: implied vol consistently exceeds realized vol (variance risk premium).
By selling options and delta-hedging, we extract this premium without directional risk.

Usage:
    python scripts/run_delta_hedge_backtest.py
    python scripts/run_delta_hedge_backtest.py --start 2023-01-01 --dte 30
    python scripts/run_delta_hedge_backtest.py --rebal-freq 5 --save

Key outputs:
    1. Daily P&L attribution (theta / gamma / vega / transaction costs)
    2. Cumulative P&L vs buy-and-hold SPY
    3. Sharpe, max drawdown, win rate
    4. IV vs RV scatter (vol risk premium visualization)
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch import fetch_historical_prices, realized_vol
from src.backtest.delta_hedge import DeltaHedgeBacktest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Delta-hedging backtest")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2023-01-01", help="Backtest start date")
    parser.add_argument("--dte", type=int, default=30, help="Days to expiry at entry")
    parser.add_argument("--rebal-freq", type=int, default=1, help="Rebalancing frequency (days)")
    parser.add_argument("--vol-model", choices=["rolling_21d", "ewm_20d", "flat"], default="rolling_21d")
    parser.add_argument("--save", action="store_true", help="Save results to CSV and figures")
    args = parser.parse_args()

    # --- 1. Data ---
    log.info(f"Loading {args.ticker} price history...")
    prices = fetch_historical_prices(args.ticker, period="3y")
    prices = prices[prices.index >= args.start]

    if len(prices) < args.dte + 10:
        raise ValueError(f"Not enough price data from {args.start}")

    spot_0 = float(prices["close"].iloc[0])
    notional = spot_0 * 100  # 1 contract = 100 shares

    # --- 2. Run backtest ---
    log.info(f"Running delta-hedge backtest: DTE={args.dte}, rebal={args.rebal_freq}d, vol={args.vol_model}")
    bt = DeltaHedgeBacktest(
        prices=prices,
        hedge_cost=0.01,
        rebal_freq=args.rebal_freq,
    )

    results = bt.run(
        entry_date=args.start,
        days_to_expiry=args.dte,
        vol_model=args.vol_model,
    )

    # --- 3. Summary stats ---
    stats = DeltaHedgeBacktest.summarize(results, notional)
    _print_summary(stats, args, notional, spot_0)

    # --- 4. Save ---
    if args.save:
        out = Path("results")
        out.mkdir(exist_ok=True)
        results.to_csv(out / "delta_hedge_results.csv", index=False)
        log.info("Saved results to results/delta_hedge_results.csv")

    # --- 5. Plots ---
    _plot_results(results, prices, args, spot_0, notional, stats)

    # --- 6. VRP analysis ---
    _plot_vrp_analysis(prices, args)

    plt.show()


def _print_summary(stats: dict, args, notional: float, spot: float):
    attr = stats["pnl_attribution"]
    attr_pct = stats["pnl_attribution_pct"]

    print("\n" + "=" * 60)
    print(f"  DELTA-HEDGE BACKTEST — {args.ticker} | DTE={args.dte} | Rebal={args.rebal_freq}d")
    print("=" * 60)
    print(f"  Total P&L:          ${stats['total_pnl']:>10,.2f}")
    print(f"  Return on Notional: {stats['return_on_notional']:>10.1%}  (notional=${notional:,.0f})")
    print(f"  Annualized Return:  {stats['annualized_return']:>10.1%}")
    print(f"  Sharpe Ratio:       {stats['sharpe']:>10.2f}")
    print(f"  Max Drawdown:       {stats['max_drawdown_pct']:>10.1%}")
    print(f"  Win Rate (daily):   {stats['win_rate']:>10.1%}")
    print(f"  Trading Days:       {stats['n_days']:>10d}")
    print()
    print("  P&L ATTRIBUTION:")
    print(f"    Theta (time decay): ${attr['theta']:>8,.2f}  ({attr_pct['theta']:>+.0%})")
    print(f"    Gamma (convexity):  ${attr['gamma']:>8,.2f}  ({attr_pct['gamma']:>+.0%})")
    print(f"    Vega  (vol moves):  ${attr['vega']:>8,.2f}  ({attr_pct['vega']:>+.0%})")
    print(f"    Transaction costs:  ${attr['transaction']:>8,.2f}")
    print("=" * 60)


def _plot_results(results, prices, args, spot_0, notional, stats):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # --- Top left: Cumulative P&L ---
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(results["day"], results["cumulative_pnl"], color="steelblue", lw=2, label="Short Straddle")
    ax1.fill_between(results["day"], results["cumulative_pnl"], 0,
                     where=results["cumulative_pnl"] >= 0, alpha=0.15, color="green")
    ax1.fill_between(results["day"], results["cumulative_pnl"], 0,
                     where=results["cumulative_pnl"] < 0, alpha=0.15, color="red")
    ax1.set_title("Cumulative P&L — Short ATM Straddle + Daily Delta Hedge")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("P&L ($)")
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Top right: IV vs daily spot vol ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(results["day"], results["iv"] * 100, s=10, alpha=0.6, color="orange", label="Implied Vol")
    rv_daily = results["dS"] / results["spot"]
    ax2.scatter(results["day"], rv_daily.abs() * np.sqrt(252) * 100, s=10, alpha=0.4, color="blue", label="Daily RV (ann)")
    ax2.set_title("IV vs Realized Vol")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Vol (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Bottom left: P&L attribution stacked ---
    ax3 = fig.add_subplot(gs[1, 0])
    components = ["cumulative_gamma", "cumulative_theta", "cumulative_vega", "cumulative_transaction"]
    labels = ["Gamma P&L", "Theta P&L", "Vega P&L", "Transaction"]
    colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for col, label, color in zip(components, labels, colors):
        ax3.plot(results["day"], results[col], label=label, color=color, lw=1.5)
    ax3.set_title("Cumulative P&L by Greek")
    ax3.set_xlabel("Days")
    ax3.set_ylabel("P&L ($)")
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # --- Bottom middle: Attribution pie ---
    ax4 = fig.add_subplot(gs[1, 1])
    attr = stats["pnl_attribution"]
    labels_pie = ["Theta", "Gamma", "Vega", "Costs"]
    values_pie = [attr["theta"], attr["gamma"], attr["vega"], attr["transaction"]]
    colors_pie = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    wedges, texts, autotexts = ax4.pie(
        [abs(v) for v in values_pie],
        labels=labels_pie, colors=colors_pie,
        autopct="%1.1f%%", startangle=90,
    )
    ax4.set_title("P&L Attribution (absolute)")

    # --- Bottom right: Daily P&L histogram ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(results["pnl_total"], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax5.axvline(results["pnl_total"].mean(), color="red", ls="--", lw=1.5,
                label=f"Mean ${results['pnl_total'].mean():.1f}")
    ax5.set_title("Daily P&L Distribution")
    ax5.set_xlabel("Daily P&L ($)")
    ax5.set_ylabel("Frequency")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.suptitle(f"{args.ticker} Short Straddle | DTE={args.dte} | Sharpe={stats['sharpe']:.2f} | MaxDD={stats['max_drawdown_pct']:.1%}",
                 fontsize=13, y=1.01)

    if args.save:
        from datetime import date
        path = Path(f"results/figures/delta_hedge_{args.ticker}_{date.today()}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved backtest figure to {path}")


def _plot_vrp_analysis(prices: pd.DataFrame, args):
    """
    Variance Risk Premium analysis: show IV - RV distribution over time.
    This is the structural edge being harvested.
    """
    rv21 = realized_vol(prices, window=21)
    rv5 = realized_vol(prices, window=5)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # VIX-proxy: assume constant IV offset for demonstration
    # In practice, fetch VIX or compute from surface
    ax = axes[0]
    ax.plot(rv21.index, rv21 * 100, label="21d Realized Vol", color="blue", lw=1.5)
    ax.plot(rv5.index, rv5 * 100, label="5d Realized Vol", color="orange", lw=1, alpha=0.7)
    ax.set_title(f"{args.ticker} Historical Realized Volatility")
    ax.set_ylabel("Volatility (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RV distribution
    ax = axes[1]
    ax.hist(rv21.dropna() * 100, bins=40, color="steelblue", edgecolor="white", alpha=0.8, density=True)
    ax.axvline(rv21.mean() * 100, color="red", ls="--", lw=1.5, label=f"Mean: {rv21.mean():.1%}")
    ax.axvline(rv21.quantile(0.75) * 100, color="orange", ls=":", lw=1.5, label=f"75th pct: {rv21.quantile(0.75):.1%}")
    ax.set_title("Realized Vol Distribution (21d)")
    ax.set_xlabel("Realized Vol (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        from datetime import date
        path = Path(f"results/figures/vrp_{args.ticker}_{date.today()}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
