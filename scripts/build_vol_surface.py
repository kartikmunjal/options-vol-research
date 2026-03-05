#!/usr/bin/env python3
"""
Build and visualize the SPY implied volatility surface.

Usage:
    python scripts/build_vol_surface.py
    python scripts/build_vol_surface.py --ticker QQQ --save
    python scripts/build_vol_surface.py --ticker SPY --plot 3d

Outputs:
    - data/processed/surface_<ticker>_<date>.parquet
    - results/figures/surface_<ticker>_<date>.png (if --save)
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch import (
    fetch_options_chain, fetch_historical_prices,
    realized_vol, fetch_vix_history, fetch_vrp_history,
)
from src.vol_surface.surface import VolSurface

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build implied vol surface")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol")
    parser.add_argument("--save", action="store_true", help="Save figures")
    parser.add_argument("--plot", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh data fetch")
    args = parser.parse_args()

    today = date.today().isoformat()
    log.info(f"Building vol surface for {args.ticker} as of {today}")

    # --- 1. Fetch Data ---
    chain, spot, r = fetch_options_chain(
        args.ticker, as_of=today, cache=not args.no_cache
    )
    prices = fetch_historical_prices(args.ticker, period="1y")
    rv = realized_vol(prices, window=21).iloc[-1]
    rv_5d = realized_vol(prices, window=5).iloc[-1]

    vix_history = fetch_vix_history(period="1y")
    vix_now = float(vix_history["vix"].iloc[-1])
    vix_regime = str(vix_history["vix_regime"].iloc[-1])

    vrp_history = fetch_vrp_history(args.ticker, period="1y")

    log.info(
        f"Spot: ${spot:.2f} | r: {r:.2%} | RV-21d: {rv:.1%} | RV-5d: {rv_5d:.1%} | "
        f"VIX: {vix_now:.1f} ({vix_regime}) | VRP: {(vix_now/100 - rv)*100:.1f} vol pts"
    )

    # --- 2. Build Surface ---
    log.info("Fitting SVI vol surface...")
    vs = VolSurface.from_chain(chain, spot=spot, r=r, build_date=today)
    log.info(f"Surface built: {len(vs.expiries)} calibrated expiries")

    # --- 3. Term Structure ---
    term_struct = vs.atm_term_structure()
    print("\n" + "=" * 60)
    print(f"  ATM TERM STRUCTURE — {args.ticker} ({today})")
    print("=" * 60)
    print(term_struct[["expiry_days", "atm_iv", "svi_rmse", "arb_free"]].to_string(index=False))

    # --- 4. Skew at 30d + VIX comparison ---
    if vs.expiries:
        T30 = min(vs.expiries, key=lambda t: abs(t - 30/365))
        skew = vs.skew(T30)
        vrp_mean = float(vrp_history["vrp"].mean())
        vrp_now = float(vrp_history["vrp"].iloc[-1])
        vrp_pct_positive = float((vrp_history["vrp"] > 0).mean())

        print(f"\n  30-DAY SKEW METRICS")
        print(f"  ATM IV (surface):  {skew['atm_iv']:.1%}")
        print(f"  VIX (SPX 30d IV):  {vix_now/100:.1%}   [regime: {vix_regime}]")
        print(f"  Risk Reversal:     {skew['risk_reversal']:.1%}  (25Δ put - 25Δ call)")
        print(f"  Butterfly:         {skew['butterfly']:.1%}  (smile convexity)")
        print(f"\n  VARIANCE RISK PREMIUM")
        print(f"  VRP today:         {vrp_now*100:+.1f} vol pts  (VIX - RV-21d)")
        print(f"  VRP 1yr mean:      {vrp_mean*100:+.1f} vol pts")
        print(f"  VRP > 0:           {vrp_pct_positive:.0%} of trading days")
        print("=" * 60)

    # --- 5. Save processed surface ---
    grid = vs.surface_grid()
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"surface_{args.ticker}_{today}.parquet"
    grid.to_parquet(out_path, index=False)
    log.info(f"Surface saved to {out_path}")

    # --- 6. Plots ---
    if args.plot in ("2d", "both"):
        _plot_smiles(vs, args.ticker, today, args.save)

    if args.plot in ("3d", "both"):
        _plot_surface_3d(vs, args.ticker, today, spot, rv, args.save)

    _plot_vrp_vs_vix(vrp_history, args.ticker, today, args.save)

    plt.show()


def _plot_smiles(vs: VolSurface, ticker: str, today: str, save: bool):
    """Plot vol smiles at multiple expiries."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, min(len(vs.expiries), 6)))
    expiries_to_plot = sorted(vs.expiries)[:6]

    # Left: IV vs Log-moneyness
    ax = axes[0]
    for T, color in zip(expiries_to_plot, colors):
        smile = vs.smile(T)
        label = f"{T*365:.0f}d"
        ax.plot(smile["log_moneyness"], smile["iv"] * 100, color=color, label=label, lw=2)

    ax.set_xlabel("Log-moneyness  k = log(K/F)")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title(f"{ticker} Volatility Smiles  [{today}]")
    ax.legend(title="Expiry", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    # Right: Term structure
    ax = axes[1]
    ts = vs.atm_term_structure()
    ax.plot(ts["expiry_days"], ts["atm_iv"] * 100, "o-", color="steelblue", lw=2, ms=6)
    ax.set_xlabel("Days to Expiry")
    ax.set_ylabel("ATM Implied Vol (%)")
    ax.set_title(f"{ticker} ATM Term Structure  [{today}]")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()

    if save:
        path = Path(f"results/figures/smiles_{ticker}_{today}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved smile plot to {path}")


def _plot_surface_3d(vs: VolSurface, ticker: str, today: str, spot: float, rv: float, save: bool):
    """3D plot of the vol surface."""
    grid = vs.surface_grid(k_range=(-0.35, 0.35), n_k=50)

    Ts = sorted(grid["T"].unique())
    Ks = sorted(grid["log_moneyness"].unique())
    Z = grid.pivot(index="T", columns="log_moneyness", values="iv").values * 100

    K_mesh, T_mesh = np.meshgrid(Ks, Ts)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_mesh, T_mesh, Z, cmap="RdYlGn_r", alpha=0.9, edgecolor="none")

    ax.set_xlabel("Log-moneyness  log(K/F)", labelpad=10)
    ax.set_ylabel("Time to Expiry (years)", labelpad=10)
    ax.set_zlabel("Implied Vol (%)", labelpad=10)
    ax.set_title(f"{ticker} Implied Volatility Surface\n{today} | Spot: ${spot:.2f} | RV-21d: {rv:.1%}")
    fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.1, label="IV (%)")

    if save:
        path = Path(f"results/figures/surface_3d_{ticker}_{today}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved 3D surface to {path}")


def _plot_vrp_vs_vix(vrp_history: pd.DataFrame, ticker: str, today: str, save: bool):
    """
    Plot VIX vs realized vol over time — the key visual for the VRP narrative.

    This is the chart that makes the strategy's edge immediately obvious:
    VIX almost always sits above realized vol, and the gap (VRP) is the
    insurance premium harvested by delta-hedged option sellers.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # --- Top: VIX vs RV ---
    ax = axes[0]
    ax.plot(vrp_history.index, vrp_history["iv_proxy"] * 100,
            label="VIX / 100 (implied vol)", color="darkorange", lw=1.8)
    ax.plot(vrp_history.index, vrp_history["rv_21d"] * 100,
            label="21d Realized Vol", color="steelblue", lw=1.5)
    ax.plot(vrp_history.index, vrp_history["rv_5d"] * 100,
            label="5d Realized Vol", color="steelblue", lw=1, alpha=0.4, ls="--")
    ax.fill_between(
        vrp_history.index,
        vrp_history["iv_proxy"] * 100,
        vrp_history["rv_21d"] * 100,
        where=(vrp_history["iv_proxy"] > vrp_history["rv_21d"]),
        alpha=0.15, color="green", label="VRP > 0 (collect premium)",
    )
    ax.fill_between(
        vrp_history.index,
        vrp_history["iv_proxy"] * 100,
        vrp_history["rv_21d"] * 100,
        where=(vrp_history["iv_proxy"] <= vrp_history["rv_21d"]),
        alpha=0.25, color="red", label="VRP < 0 (realize > implied)",
    )
    ax.set_ylabel("Volatility (%)")
    ax.set_title(f"{ticker}: VIX vs Realized Vol — The Variance Risk Premium  [{today}]")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # VIX regime bands
    ax.axhline(15, color="green", lw=0.7, ls=":", alpha=0.6)
    ax.axhline(25, color="orange", lw=0.7, ls=":", alpha=0.6)
    ax.axhline(40, color="red", lw=0.7, ls=":", alpha=0.6)

    # --- Bottom: VRP time series ---
    ax = axes[1]
    vrp_pct = vrp_history["vrp"] * 100
    ax.bar(
        vrp_history.index, vrp_pct,
        color=vrp_pct.apply(lambda v: "seagreen" if v > 0 else "crimson"),
        alpha=0.7, width=1.5,
    )
    ax.axhline(vrp_pct.mean(), color="black", lw=1.5, ls="--",
               label=f"Mean VRP: {vrp_pct.mean():.1f} vol pts")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("VRP (vol points)")
    ax.set_title("Variance Risk Premium = VIX − 21d RV")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = Path(f"results/figures/vrp_vs_vix_{ticker}_{today}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved VRP chart to {path}")


if __name__ == "__main__":
    main()
