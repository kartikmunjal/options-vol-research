# Options Volatility Research

**Core question:** Can the variance risk premium — the systematic spread between implied and realized volatility — be harvested through delta-hedged options strategies?

## Background

Implied volatility for SPY has historically traded **2–4 vol points above realized volatility** on average. This "insurance premium" exists because market participants pay to own options as portfolio protection, creating a persistent structural edge for systematic option sellers.

This project builds the full quantitative infrastructure to:
1. Construct and analyze the complete implied volatility surface
2. Decompose P&L from delta-hedged positions into its Greek components
3. Quantify the variance risk premium and its regime dependence

## Architecture

```
src/
├── pricing/
│   ├── black_scholes.py     # Hand-rolled BS pricing + 8 Greeks (no library calls)
│   └── implied_vol.py       # Newton-Raphson IV solver with bisection fallback
├── vol_surface/
│   ├── svi.py               # SVI parametrization (Gatheral 2004) + arbitrage checks
│   └── surface.py           # Full surface construction pipeline
├── data/
│   └── fetch.py             # yfinance options chain + historical prices + FRED rates
└── backtest/
    └── delta_hedge.py       # Daily delta-hedging engine + full P&L attribution
```

## Key Design Decisions

**Why SVI?** The Stochastic Volatility Inspired parametrization (Gatheral, 2004) gives a globally arbitrage-free surface with just 5 parameters per expiry slice. It outperforms cubic splines (can produce negative densities) and Heston (computationally heavier, harder to calibrate to market quotes directly). SVI is used in production at major vol desks.

**Why hand-roll Greeks?** Every Greek in `black_scholes.py` is derived analytically from the BS PDE solution — no numerical differentiation, no library abstraction. This matters in production where Vega computation speed is a bottleneck for calibration.

**Why Newton-Raphson for IV?** NR converges quadratically near the solution (Vega ≠ 0), typically finishing in 3–5 iterations for near-ATM options. The bisection fallback handles deep OTM strikes where Vega ≈ 0 and NR can overshoot into negative vol territory.

## P&L Attribution Framework

For a short straddle with daily delta hedging, P&L decomposes as:

```
ΔP&L ≈ Gamma P&L + Theta P&L + Vega P&L + Transaction Costs

where:
  Gamma P&L  = -½ Γ S² (ΔS/S)²      [negative = short gamma loses on large moves]
  Theta P&L  = -Θ · Δt               [positive = collect time decay daily]
  Vega P&L   = -ν · Δσ               [negative if vol expands]
```

At expiry, the net P&L from delta-hedging converges to:

```
P&L_total ≈ ½ · ∫ Γ S² (σ_IV² - σ_RV²) dt
```

This integral is positive when **IV > RV** — the core bet.

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/options-vol-research.git
cd options-vol-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build today's SPY vol surface
python scripts/build_vol_surface.py --save

# Run delta-hedge backtest
python scripts/run_delta_hedge_backtest.py --start 2023-01-01 --dte 30 --save
```

## Results

_Run the scripts to generate live results. Example output structure:_

```
ATM TERM STRUCTURE — SPY (2026-03-04)
 expiry_days  atm_iv  svi_rmse
           7   0.142     0.003
          14   0.148     0.002
          30   0.155     0.002
          60   0.162     0.003
         180   0.171     0.004

DELTA-HEDGE BACKTEST (DTE=30, daily rebal)
  Annualized Return:   18.3%
  Sharpe Ratio:         1.41
  Max Drawdown:        -9.2%
  Win Rate (daily):    61.4%

P&L ATTRIBUTION:
  Theta:        +$4,210  (+73%)   [collected daily time decay]
  Gamma:        -$2,140  (-37%)   [cost of large spot moves]
  Vega:         -$ 890  (-15%)   [cost of vol expansion]
  Costs:        -$ 180   (-3%)   [transaction costs]
```

## References

- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parametrization.*
- Carr, P. & Wu, L. (2009). *Variance Risk Premiums.* Review of Financial Studies.
- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.*
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
