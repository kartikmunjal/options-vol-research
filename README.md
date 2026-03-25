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

Live results from `python scripts/run_delta_hedge_backtest.py --dte 30`:

```
ATM TERM STRUCTURE — SPY (3-year average, VIX proxy + SVI fit)
 expiry_days  atm_iv  svi_rmse
           7   0.082     0.0036
          14   0.116     0.0029
          30   0.170     0.0021
          60   0.241     0.0016
         180   0.417     0.0017

DELTA-HEDGE BACKTEST — SPY Short ATM Straddle
  Period:             2023-05-10 to 2026-03-25  (25 trades × 30 DTE, daily rebal)
  Notional:           $39,544  (1 contract @ entry)
  Total P&L:          $17,685
  Return on Notional:  44.7%
  Annualized Return:   15.6%
  Sharpe Ratio:         1.02
  Max Drawdown:        -14.0%
  Win Rate (daily):    56.7%

P&L ATTRIBUTION:
  Theta (time decay):  +$21,506  (+122%)  [structural edge — IV > RV]
  Gamma (convexity):   -$24,556  (-139%)  [cost of spot moves]
  Vega  (vol moves):   +$19,366  (+110%)  [VRP mean-reversion benefit]
  Transaction costs:       -$83

VARIANCE RISK PREMIUM (SPY, VIX − RV_21d, 3-year sample):
  Mean VIX:             17.0 pts
  Mean RV (21d):        13.3 pts
  Mean VRP:              3.7 pts  [IV trades ~3.7 vol points above realized vol]
  VRP > 0:              89.5% of trading days
```

The large positive theta (+122%) and negative gamma (-139%) confirm the variance risk
premium mechanism: implied vol (VIX avg 17.0) systematically exceeded realized vol
(13.3), yielding a structural 3.7 vol-point edge on 89.5% of days. Vega is net
positive over this period because vol contracted on balance (VRP mean-reversion).
The Sharpe of 1.02 on a simple 1-contract strategy with 1¢ transaction costs is
net-realistic.

## C++ Acceleration (`cpp/`)

The `cpp/` directory is a drop-in high-performance rewrite of the pricing core in C++17,
exposing a `vol_core` Python extension via **pybind11**. It is a strict superset of the
Python layer: same formulas, same conventions, ~10–50× faster.

### What's in `cpp/`

```
cpp/
├── src/
│   ├── black_scholes.hpp   # Templated BS pricer + all 8 Greeks (single-pass)
│   ├── iv_solver.hpp       # Newton-Raphson + bisection fallback; parallel strip solve
│   ├── svi.hpp             # SVI surface: analytic gradient, arbitrage checks, calibration
│   └── bindings.cpp        # pybind11 module (py::vectorize for numpy compat)
├── tests/
│   └── test_greeks.cpp     # Catch2: put-call parity, FD verification, IV round-trip, SVI
├── benchmarks/
│   ├── bench_surface.cpp   # Google Benchmark microbenchmarks
│   └── bench_python.py     # Python-side timing vs scipy reference
├── CMakeLists.txt
└── setup.py
```

### Build

**Option A — pip (recommended for notebook use):**
```bash
cd cpp
pip install -e .
cd ..
python -c "import vol_core; print(vol_core.__version__)"
```

**Option B — CMake (also builds C++ tests and benchmarks):**
```bash
cmake -B cpp/build -DCMAKE_BUILD_TYPE=Release cpp
cmake --build cpp/build -j$(nproc)
# Run tests:
./cpp/build/test_greeks
# Run benchmarks:
./cpp/build/bench_surface --benchmark_format=table
```

Requirements: C++17 compiler (GCC ≥ 9 / Clang ≥ 10 / MSVC 2019+), CMake ≥ 3.15, Python ≥ 3.8.
pybind11, Catch2, and Google Benchmark are fetched automatically via `FetchContent`.

### API

```python
import vol_core as vc
import numpy as np

# Scalar pricing
vc.bs_price(100, 100, 1.0, 0.05, 0.20, True)           # ~10.4506

# Vectorized over strikes (numpy array)
strikes = np.linspace(80, 120, 41)
prices  = vc.bs_price(100, strikes, 1.0, 0.05, 0.20, True)

# All 8 Greeks in a single pass (same cost as pricing alone)
g = vc.bs_all_greeks(100, 100, 1.0, 0.05, 0.20, True)
# → dict: price, delta, gamma, theta, vega, vanna, volga, charm, rho

# IV solver: Newton-Raphson + bisection fallback
iv = vc.implied_vol(10.45, 100, 100, 1.0, 0.05, True)  # → 0.20

# Parallel IV strip (C++17 std::execution::par_unseq)
ivs = vc.implied_vol_strip(prices, strikes, 100, 1.0, 0.05, True)

# SVI calibration
res  = vc.calibrate_svi(k_arr, iv_arr, T=0.5)   # → {a, b, rho, m, sigma, rmse, arb_check}
surf = vc.SVISurface()
surf.add_slice(T=0.5, **{k: res[k] for k in ['a','b','rho','m','sigma']})
surf.is_arbitrage_free()   # → True / False
```

### Performance (indicative, M1 Pro)

| Workload | Python (scipy) | C++ (`vol_core`) | Speedup |
|----------|---------------|-----------------|---------|
| Single BS price | ~6 µs | ~0.15 µs | **40×** |
| All 8 Greeks | ~20 µs | ~0.3 µs | **65×** |
| Price 200 strikes | ~1.2 ms | ~0.04 ms | **30×** |
| IV strip N=100 | ~80 ms | ~3 ms | **25×** |
| SVI calibrate (11 pts) | ~2.5 ms | ~0.15 ms | **17×** |

Run `python cpp/benchmarks/bench_python.py` for machine-specific numbers.

### Key Engineering Decisions

**`BSIntermediates<T>` shared precomputation** — d1, d2, Φ(d1), Φ(d2), φ(d1), Se^{−qT}, Ke^{−rT}
computed once; all 8 Greeks derived from the same structure. Net cost: same as calling
`price()` alone, regardless of how many Greeks are requested.

**Machine-precision `ncdf`** — `0.5 × erfc(−x / √2)` maps exactly to a hardware FMA+erfc
instruction on modern CPUs. Avoids the series-expansion approximations in Python's `scipy.stats.norm.cdf`.

**IV Newton-Raphson** — reuses `BSIntermediates` inside each NR step to get both price
and raw vega at the cost of a single transcendental evaluation. Bisection fallback
on [1e-4, 5.0] guarantees convergence.

**SVI analytic gradient** — closed-form ∂w/∂θ for all 5 SVI parameters eliminates
finite-difference overhead. Combined with Armijo backtracking and 4 random restarts,
calibration is both fast and robust to local minima.

## References

- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parametrization.*
- Carr, P. & Wu, L. (2009). *Variance Risk Premiums.* Review of Financial Studies.
- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.*
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
- Brenner, M. & Subrahmanyam, M. (1988). *A simple formula to compute the implied standard deviation.*
