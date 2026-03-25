"""
bench_python.py  —  Python-side timing for head-to-head C++ vs Python comparison

Runs the same workloads as bench_surface.cpp using:
  - vol_core  : C++ extension (must be compiled first)
  - Pure Python: scipy.stats + numpy implementations

Usage:
  # Build the C++ extension first:
  cd cpp && pip install -e . && cd ..
  python cpp/benchmarks/bench_python.py

Expected speedup:
  BS pricing:     ~20-50x  (C++ avoids Python overhead, uses inlined math)
  All 8 Greeks:   ~25-60x  (single-pass C++ vs 8 separate Python calls)
  IV strip N=100: ~15-40x  (parallel C++ vs vectorized scipy loop)
  SVI calibrate:  ~10-20x  (compiled gradient descent vs Python loop)
"""

import sys
import time
import textwrap
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────────────────────
# Pure Python (numpy) Black-Scholes reference implementations
# ─────────────────────────────────────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma, q=0.0):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2

def py_bs_price(S, K, T, r, sigma, is_call, q=0.0):
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def py_bs_all_greeks(S, K, T, r, sigma, is_call, q=0.0):
    """Compute all 8 Greeks in Python — multiple norm.cdf/pdf calls."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    nd1  = norm.cdf(d1 if is_call else -d1)
    nd2  = norm.cdf(d2 if is_call else -d2)
    phi  = norm.pdf(d1)
    sign = 1 if is_call else -1

    Se_qT = S * np.exp(-q * T)
    Ke_rT = K * np.exp(-r * T)

    price = sign * (Se_qT * norm.cdf(sign * d1) - Ke_rT * norm.cdf(sign * d2))
    delta = sign * np.exp(-q * T) * norm.cdf(sign * d1)
    gamma = np.exp(-q * T) * phi / (S * sigma * sqrt_T)
    vega  = Se_qT * phi * sqrt_T * 0.01
    theta = (-(Se_qT * phi * sigma) / (2 * sqrt_T)
             - sign * (r * Ke_rT * norm.cdf(sign * d2)
                       - q * Se_qT * norm.cdf(sign * d1))) / 365.0
    vanna = -np.exp(-q * T) * phi * d2 / sigma
    volga = Se_qT * phi * sqrt_T * d1 * d2 / sigma
    charm_raw = -np.exp(-q * T) * phi * (
        2 * (r - q) * T - d2 * sigma * sqrt_T
    ) / (2 * T * sigma * sqrt_T)
    charm = sign * charm_raw / 365.0 if not is_call else charm_raw / 365.0
    rho   = sign * Ke_rT * T * norm.cdf(sign * d2) * 1e-4

    return dict(price=price, delta=delta, gamma=gamma, theta=theta,
                vega=vega, vanna=vanna, volga=volga, charm=charm, rho=rho)

def py_implied_vol(market_price, S, K, T, r, is_call, q=0.0):
    """Brentq IV solver (scipy)."""
    intrinsic = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T)) if is_call else \
                max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))
    if market_price < intrinsic:
        return float('nan')
    try:
        return brentq(
            lambda v: py_bs_price(S, K, T, r, v, is_call, q) - market_price,
            1e-4, 5.0, xtol=1e-8, full_output=False
        )
    except Exception:
        return float('nan')

def py_iv_strip(market_prices, strikes, S, T, r, is_call, q=0.0):
    return [py_implied_vol(p, S, k, T, r, is_call, q)
            for p, k in zip(market_prices, strikes)]

# ─────────────────────────────────────────────────────────────────────────────
# Timer helper
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager + repeat timer with warm-up."""

    def __init__(self, n_warmup=3, n_repeat=10):
        self.n_warmup = n_warmup
        self.n_repeat = n_repeat

    def timeit(self, fn, *args, **kwargs):
        for _ in range(self.n_warmup):
            fn(*args, **kwargs)
        times = []
        for _ in range(self.n_repeat):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)
        return np.array(times)

    @staticmethod
    def fmt(times_sec, n_items=1):
        mean_us = np.mean(times_sec) * 1e6
        std_us  = np.std(times_sec)  * 1e6
        per_item_us = mean_us / n_items
        return f"{mean_us:8.1f} ± {std_us:.1f} µs  ({per_item_us:.3f} µs/item)"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark suite
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmarks(vc):
    """Run all benchmarks and print a comparison table."""
    timer = Timer(n_warmup=5, n_repeat=20)

    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

    print("\n" + "="*80)
    print("  vol_core C++ vs Python Benchmark  (all times: mean ± std, µs)")
    print("="*80)
    print(f"  {'Benchmark':<40}  {'Python':>20}  {'C++':>20}  {'Speedup':>10}")
    print("-"*80)

    rows = []

    # 1. Single BS price
    t_py = timer.timeit(py_bs_price, S, K, T, r, sigma, True, q)
    t_cc = timer.timeit(vc.bs_price, S, K, T, r, sigma, True, q)
    rows.append(("BS price (single)", t_py, t_cc, 1))

    # 2. All 8 Greeks (single pass vs 8 Python calls)
    t_py = timer.timeit(py_bs_all_greeks, S, K, T, r, sigma, True, q)
    t_cc = timer.timeit(vc.bs_all_greeks, S, K, T, r, sigma, True, q)
    rows.append(("All 8 Greeks", t_py, t_cc, 1))

    # 3. Vectorised pricing: 200-strike grid
    strikes_200 = np.linspace(70, 130, 200)
    # Python: loop
    t_py = timer.timeit(lambda: np.array([py_bs_price(S, k, T, r, sigma, True, q)
                                          for k in strikes_200]))
    t_cc = timer.timeit(vc.bs_price, S, strikes_200, T, r, sigma, True, q)
    rows.append(("BS price grid N=200", t_py, t_cc, len(strikes_200)))

    # 4. Single IV solve
    market_price = py_bs_price(S, K, T, r, sigma, True, q)
    t_py = timer.timeit(py_implied_vol, market_price, S, K, T, r, True, q)
    t_cc = timer.timeit(vc.implied_vol, market_price, S, K, T, r, True, q)
    rows.append(("Implied vol (single)", t_py, t_cc, 1))

    # 5. IV strip: N=41, N=201
    for N in (41, 201):
        strikes_N = np.linspace(70, 130, N)
        prices_N  = np.array([py_bs_price(S, k, T, r, sigma, True, 0.0) for k in strikes_N])
        t_py = timer.timeit(py_iv_strip, prices_N, strikes_N, S, T, r, True, 0.0)
        t_cc = timer.timeit(vc.implied_vol_strip, prices_N, strikes_N, S, T, r, True)
        rows.append((f"IV strip N={N}", t_py, t_cc, N))

    # 6. SVI calibration on 11-strike smile
    svi_k  = np.linspace(-0.3, 0.3, 11)
    svi_iv = np.array([vc.svi_implied_vol(k, 0.5, 0.04, 0.10, -0.30, 0.0, 0.15)
                       for k in svi_k])
    t_cc = timer.timeit(vc.calibrate_svi, svi_k, svi_iv, 0.5)
    # Python SVI calibration: use scipy.optimize.minimize as reference
    try:
        from scipy.optimize import minimize

        def svi_w(k, a, b, rho, m, sigma):
            disc = np.sqrt((k - m)**2 + sigma**2)
            return a + b * (rho * (k - m) + disc)

        def svi_obj(params):
            a, b, rho, m, sigma = params
            w_fit = svi_w(svi_k, a, b, rho, m, sigma)
            iv_fit = np.sqrt(np.maximum(w_fit, 0) / 0.5)
            return np.sum((iv_fit - svi_iv)**2)

        x0 = [0.04, 0.10, -0.3, 0.0, 0.15]
        t_py = timer.timeit(lambda: minimize(svi_obj, x0, method='L-BFGS-B',
                                             options={'maxiter': 1000}))
        rows.append(("SVI calibrate (11 strikes)", t_py, t_cc, 1))
    except ImportError:
        rows.append(("SVI calibrate (11 strikes)", None, t_cc, 1))

    # Print table
    for name, t_py, t_cc, n_items in rows:
        cc_str = timer.fmt(t_cc, n_items)
        if t_py is not None:
            py_str  = timer.fmt(t_py, n_items)
            speedup = np.mean(t_py) / np.mean(t_cc)
            print(f"  {name:<40}  {py_str:>20}  {cc_str:>20}  {speedup:>8.1f}x")
        else:
            print(f"  {name:<40}  {'(N/A)':>20}  {cc_str:>20}  {'—':>10}")

    print("="*80)
    print("\nNotes:")
    print("  - C++ timing includes Python→C++ argument conversion overhead")
    print("  - IV strip uses C++17 std::execution::par_unseq (parallel)")
    print("  - SVI C++: 4 restarts, analytic gradient + Armijo line search")
    print("  - SVI Python: scipy L-BFGS-B (no restarts, numerical gradient)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Numerical correctness check (quick sanity before benchmarking)
# ─────────────────────────────────────────────────────────────────────────────

def check_correctness(vc):
    print("\nSanity checks:")
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

    # ATM call
    py_p  = py_bs_price(S, K, T, r, sigma, True, q)
    cpp_p = vc.bs_price(S, K, T, r, sigma, True, q)
    assert abs(py_p - cpp_p) < 1e-8, f"Price mismatch: {py_p} vs {cpp_p}"
    print(f"  ATM call price: Python={py_p:.6f}, C++={float(cpp_p):.6f}  ✓")

    # Put-call parity
    call = vc.bs_price(S, K, T, r, sigma, True,  q)
    put  = vc.bs_price(S, K, T, r, sigma, False, q)
    pcp  = float(call) - float(put) - (S * np.exp(-q * T) - K * np.exp(-r * T))
    assert abs(pcp) < 1e-9, f"PCP violation: {pcp}"
    print(f"  Put-call parity error: {pcp:.2e}  ✓")

    # IV round-trip
    market_p = float(vc.bs_price(S, K, T, r, sigma, True, q))
    iv = vc.implied_vol(market_p, S, K, T, r, True, q)
    repriced = vc.bs_price(S, K, T, r, float(iv), True, q)
    iv_err = abs(float(repriced) - market_p) / market_p
    assert iv_err < 1e-7, f"IV round-trip error: {iv_err:.2e}"
    print(f"  IV round-trip relative error: {iv_err:.2e}  ✓")

    # All 8 Greeks present
    g = vc.bs_all_greeks(S, K, T, r, sigma, True, q)
    for key in ('price', 'delta', 'gamma', 'theta', 'vega', 'vanna', 'volga', 'charm', 'rho'):
        assert key in g, f"Missing Greek: {key}"
    print(f"  All 8 Greeks present  ✓")

    # SVI calibration
    k_arr  = np.linspace(-0.3, 0.3, 11)
    iv_arr = np.array([vc.svi_implied_vol(k, 0.5, 0.04, 0.10, -0.30, 0.0, 0.15) for k in k_arr])
    res = vc.calibrate_svi(k_arr, iv_arr, 0.5)
    assert res['rmse'] < 1e-5, f"SVI RMSE too large: {res['rmse']:.2e}"
    assert res['converged'], "SVI calibration did not converge"
    print(f"  SVI calibration RMSE: {res['rmse']:.2e}  converged={res['converged']}  ✓")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Try to import the compiled extension
    try:
        # Add cpp/ to path so we can find vol_core.so
        cpp_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(cpp_dir))
        import vol_core as vc
        print(f"Loaded vol_core version: {vc.__version__}")
    except ImportError as e:
        print(f"\nERROR: Could not import vol_core: {e}")
        print("\nBuild the extension first:")
        print("  cd cpp && pip install -e . && cd ..")
        print("  # or:")
        print("  cmake -B cpp/build -DCMAKE_BUILD_TYPE=Release cpp && cmake --build cpp/build")
        sys.exit(1)

    check_correctness(vc)
    run_benchmarks(vc)
