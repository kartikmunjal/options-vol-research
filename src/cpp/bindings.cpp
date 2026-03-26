/**
 * bindings.cpp  —  pybind11 Python bindings for vol_core module
 *
 * Exposes:
 *   - Black-Scholes pricing and all 8 Greeks (vectorized via py::vectorize)
 *   - Newton-Raphson / bisection IV solver (single + strip)
 *   - SVI calibration and surface class
 *
 * Build:
 *   pip install ./cpp              # uses setup.py
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
 *
 * Usage (Python):
 *   import vol_core as vc
 *   vc.bs_price(100, 100, 1.0, 0.05, 0.20, True)     # call price
 *   vc.implied_vol(10.45, 100, 100, 1.0, 0.05, True)  # IV from market price
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "black_scholes.hpp"
#include "iv_solver.hpp"
#include "svi.hpp"

namespace py = pybind11;
using namespace volcpp;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: wrap Greeks<double> → Python dict
// ─────────────────────────────────────────────────────────────────────────────
static py::dict greeks_to_dict(const Greeks<double>& g) {
    py::dict d;
    d["price"] = g.price;
    d["delta"] = g.delta;
    d["gamma"] = g.gamma;
    d["theta"] = g.theta;   // per calendar day
    d["vega"]  = g.vega;    // per vol point (0.01)
    d["vanna"] = g.vanna;
    d["volga"] = g.volga;
    d["charm"] = g.charm;   // per calendar day
    d["rho"]   = g.rho;     // per bp (0.0001)
    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar + vectorised BS wrappers
// ─────────────────────────────────────────────────────────────────────────────

// Scalar all-greeks — returns dict
static py::dict py_all_greeks(double S, double K, double T, double r,
                               double sigma, bool is_call, double q = 0.0) {
    return greeks_to_dict(BlackScholes<double>::all_greeks(S, K, T, r, sigma, is_call, q));
}

// Vectorised pricing — accepts numpy arrays or scalars
auto vbs_price = py::vectorize(
    [](double S, double K, double T, double r, double sigma, bool is_call, double q) {
        return BlackScholes<double>::price(S, K, T, r, sigma, is_call, q);
    });

auto vbs_delta = py::vectorize(
    [](double S, double K, double T, double r, double sigma, bool is_call, double q) {
        return BlackScholes<double>::delta(S, K, T, r, sigma, is_call, q);
    });

auto vbs_gamma = py::vectorize(
    [](double S, double K, double T, double r, double sigma, double q) {
        return BlackScholes<double>::gamma(S, K, T, r, sigma, q);
    });

auto vbs_theta = py::vectorize(
    [](double S, double K, double T, double r, double sigma, bool is_call, double q) {
        return BlackScholes<double>::theta(S, K, T, r, sigma, is_call, q);
    });

auto vbs_vega = py::vectorize(
    [](double S, double K, double T, double r, double sigma, double q) {
        return BlackScholes<double>::vega(S, K, T, r, sigma, q);
    });

auto vbs_vanna = py::vectorize(
    [](double S, double K, double T, double r, double sigma, double q) {
        return BlackScholes<double>::vanna(S, K, T, r, sigma, q);
    });

auto vbs_volga = py::vectorize(
    [](double S, double K, double T, double r, double sigma, double q) {
        return BlackScholes<double>::volga(S, K, T, r, sigma, q);
    });

auto vbs_charm = py::vectorize(
    [](double S, double K, double T, double r, double sigma, bool is_call, double q) {
        return BlackScholes<double>::charm(S, K, T, r, sigma, is_call, q);
    });

auto vbs_rho = py::vectorize(
    [](double S, double K, double T, double r, double sigma, bool is_call, double q) {
        return BlackScholes<double>::rho(S, K, T, r, sigma, is_call, q);
    });

// ─────────────────────────────────────────────────────────────────────────────
// IV solver wrappers
// ─────────────────────────────────────────────────────────────────────────────

static double py_implied_vol(double market_price, double S, double K,
                             double T, double r, bool is_call, double q = 0.0,
                             double tol = 1e-8) {
    IVSolver<double>::Options opts;
    opts.tol = tol;
    return IVSolver<double>::solve(market_price, S, K, T, r, is_call, q, opts);
}

// Strip: parallel solve over a vector of (market_price, strike) pairs
static py::array_t<double> py_implied_vol_strip(
    py::array_t<double, py::array::c_style | py::array::forcecast> market_prices,
    py::array_t<double, py::array::c_style | py::array::forcecast> strikes,
    double S, double T, double r, bool is_call, double q = 0.0, double tol = 1e-8)
{
    auto buf_p = market_prices.request();
    auto buf_k = strikes.request();
    if (buf_p.size != buf_k.size)
        throw std::invalid_argument("market_prices and strikes must have the same length");

    std::vector<double> vp(static_cast<double*>(buf_p.ptr),
                           static_cast<double*>(buf_p.ptr) + buf_p.size);
    std::vector<double> vk(static_cast<double*>(buf_k.ptr),
                           static_cast<double*>(buf_k.ptr) + buf_k.size);

    IVSolver<double>::Options opts;
    opts.tol = tol;
    std::vector<double> result = IVSolver<double>::solve_strip(vp, vk, S, T, r, is_call, q, opts);

    py::array_t<double> out(result.size());
    std::copy(result.begin(), result.end(), static_cast<double*>(out.request().ptr));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// SVI wrappers
// ─────────────────────────────────────────────────────────────────────────────

static double py_svi_total_variance(double k, double a, double b, double rho,
                                    double m, double sigma) {
    return SVI::total_variance(k, {a, b, rho, m, sigma});
}

static double py_svi_implied_vol(double k, double T,
                                 double a, double b, double rho,
                                 double m, double sigma) {
    return SVI::implied_vol(k, T, {a, b, rho, m, sigma});
}

// Vectorised SVI implied vol (over k-grid)
static py::array_t<double> py_svi_vol_curve(
    py::array_t<double, py::array::c_style | py::array::forcecast> k_arr,
    double T, double a, double b, double rho, double m, double sigma)
{
    SVIParams p{a, b, rho, m, sigma};
    auto buf = k_arr.request();
    const double* kp = static_cast<const double*>(buf.ptr);
    py::array_t<double> out(buf.size);
    double* op = static_cast<double*>(out.request().ptr);
    for (ssize_t i = 0; i < buf.size; ++i)
        op[i] = SVI::implied_vol(kp[i], T, p);
    return out;
}

// Calibrate SVI slice: returns dict {a, b, rho, m, sigma, rmse, converged}
static py::dict py_calibrate_svi(
    py::array_t<double, py::array::c_style | py::array::forcecast> k_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> iv_arr,
    double T, int n_restarts = 4, double lambda_arb = 1e5)
{
    auto bk = k_arr.request();
    auto bv = iv_arr.request();
    if (bk.size != bv.size)
        throw std::invalid_argument("k and iv arrays must have the same length");

    std::vector<double> k_vec(static_cast<double*>(bk.ptr),
                              static_cast<double*>(bk.ptr) + bk.size);
    std::vector<double> iv_vec(static_cast<double*>(bv.ptr),
                               static_cast<double*>(bv.ptr) + bv.size);

    auto res = SVI::calibrate(k_vec, iv_vec, T, n_restarts, lambda_arb);

    py::dict d;
    d["a"]         = res.params.a;
    d["b"]         = res.params.b;
    d["rho"]       = res.params.rho;
    d["m"]         = res.params.m;
    d["sigma"]     = res.params.sigma;
    d["rmse"]      = res.rmse;
    d["converged"] = res.converged;
    d["n_iter"]    = res.n_iter;

    // Arbitrage checks
    py::dict arb;
    arb["butterfly_ok"]    = res.arb_check.butterfly_ok;
    arb["min_variance_ok"] = res.arb_check.min_variance_ok;
    arb["butterfly_slack"] = res.arb_check.butterfly_slack;
    arb["min_variance"]    = res.arb_check.min_variance;
    d["arb_check"] = arb;

    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// SVISurface Python class
// ─────────────────────────────────────────────────────────────────────────────

struct PySVISurface {
    SVISurface surface;

    void add_slice(double T, double a, double b, double rho, double m, double sigma) {
        SVISlice sl;
        sl.T      = T;
        sl.params = {a, b, rho, m, sigma};
        surface.add_slice(std::move(sl));
    }

    double implied_vol(double K, double T) const {
        return surface.implied_vol(K, T);
    }

    // Returns list of (T, is_ok) tuples
    py::list check_calendar() const {
        auto checks = surface.check_calendar();
        py::list out;
        for (auto& [T, ok] : checks)
            out.append(py::make_tuple(T, ok));
        return out;
    }

    bool is_arbitrage_free() const {
        return surface.is_fully_arbitrage_free();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(vol_core, m) {
    m.doc() = R"doc(
    vol_core — C++ accelerated volatility surface library

    Black-Scholes pricing with all 8 Greeks (analytic), Newton-Raphson IV solver
    with bisection fallback, and SVI surface construction with no-arbitrage checks.

    All pricing functions accept both scalars and numpy arrays.

    Example
    -------
    >>> import vol_core as vc
    >>> import numpy as np
    >>> vc.bs_price(100, 100, 1.0, 0.05, 0.20, True)   # ~10.4506
    >>> strikes = np.linspace(80, 120, 41)
    >>> prices  = vc.bs_price(100, strikes, 1.0, 0.05, 0.20, True)
    >>> ivs = vc.implied_vol_strip(prices, strikes, 100, 1.0, 0.05, True)
    )doc";

    // ── Black-Scholes ────────────────────────────────────────────────────────
    m.def("bs_price", vbs_price,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("is_call"), py::arg("q") = 0.0,
          R"doc(Black-Scholes option price (vectorized).

Parameters
----------
S, K      : spot and strike
T         : time to expiry (years)
r         : risk-free rate (continuous)
sigma     : implied volatility (e.g. 0.20 = 20%)
is_call   : True=call, False=put
q         : continuous dividend yield (default 0)

Returns
-------
Option price. NaN for invalid inputs (T<=0, sigma<=0).
)doc");

    m.def("bs_delta", vbs_delta,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("is_call"), py::arg("q") = 0.0,
          "Delta: dP/dS. Call ∈ (0,1), Put ∈ (-1,0).");

    m.def("bs_gamma", vbs_gamma,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("q") = 0.0,
          "Gamma: d²P/dS². Same for calls and puts.");

    m.def("bs_theta", vbs_theta,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("is_call"), py::arg("q") = 0.0,
          "Theta: dP/dt per calendar day (negative for long options).");

    m.def("bs_vega", vbs_vega,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("q") = 0.0,
          "Vega: dP/dσ per 1 vol point (0.01). Same for calls and puts.");

    m.def("bs_vanna", vbs_vanna,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("q") = 0.0,
          "Vanna: d²P/dS dσ = dDelta/dσ.");

    m.def("bs_volga", vbs_volga,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("q") = 0.0,
          "Volga (vomma): d²P/dσ². Curvature of price in vol space.");

    m.def("bs_charm", vbs_charm,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("is_call"), py::arg("q") = 0.0,
          "Charm: dDelta/dt per calendar day. Also called delta decay.");

    m.def("bs_rho", vbs_rho,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("is_call"), py::arg("q") = 0.0,
          "Rho: dP/dr per basis point (0.0001).");

    m.def("bs_all_greeks", &py_all_greeks,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("is_call"), py::arg("q") = 0.0,
          R"doc(Compute all 8 Greeks in a single pass (same cost as pricing alone).

Returns
-------
dict with keys: price, delta, gamma, theta, vega, vanna, volga, charm, rho
)doc");

    // ── IV Solver ────────────────────────────────────────────────────────────
    m.def("implied_vol", &py_implied_vol,
          py::arg("market_price"), py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("is_call"), py::arg("q") = 0.0,
          py::arg("tol") = 1e-8,
          R"doc(Compute implied volatility via Newton-Raphson + bisection fallback.

Returns NaN if the market price is outside the no-arbitrage bounds.
Precision: ~1e-10 typical; guaranteed to tol via bisection fallback.
)doc");

    m.def("implied_vol_strip", &py_implied_vol_strip,
          py::arg("market_prices"), py::arg("strikes"),
          py::arg("S"), py::arg("T"), py::arg("r"), py::arg("is_call"),
          py::arg("q") = 0.0, py::arg("tol") = 1e-8,
          R"doc(Compute IV for a strip of options (same expiry, varying strikes).

Uses C++17 parallel execution (std::execution::par_unseq) when available.

Parameters
----------
market_prices : array of option prices
strikes       : array of strikes (same length)
S, T, r       : spot, expiry, rate
is_call       : all options have the same flag (use put-call parity to mix)
q             : continuous dividend yield

Returns
-------
numpy array of IVs (NaN where solution not found)
)doc");

    // ── SVI ──────────────────────────────────────────────────────────────────
    m.def("svi_total_variance", &py_svi_total_variance,
          py::arg("k"), py::arg("a"), py::arg("b"), py::arg("rho"),
          py::arg("m"), py::arg("sigma"),
          "SVI total variance w(k) = a + b[ρ(k-m) + √((k-m)²+σ²)].");

    m.def("svi_implied_vol", &py_svi_implied_vol,
          py::arg("k"), py::arg("T"), py::arg("a"), py::arg("b"),
          py::arg("rho"), py::arg("m"), py::arg("sigma"),
          "SVI implied vol: √(w(k)/T). k = log(K/F) is log-moneyness.");

    m.def("svi_vol_curve", &py_svi_vol_curve,
          py::arg("k"), py::arg("T"), py::arg("a"), py::arg("b"),
          py::arg("rho"), py::arg("m"), py::arg("sigma"),
          "SVI implied vol curve over a k-grid (numpy array). Returns numpy array.");

    m.def("calibrate_svi", &py_calibrate_svi,
          py::arg("k"), py::arg("iv"), py::arg("T"),
          py::arg("n_restarts") = 4, py::arg("lambda_arb") = 1e5,
          R"doc(Calibrate SVI parameters to market implied vols.

Parameters
----------
k          : log-moneyness array log(K/F)
iv         : market implied vol array (e.g. 0.20 = 20%)
T          : time to expiry (years)
n_restarts : number of random restarts (default 4)
lambda_arb : butterfly penalty weight (default 1e5)

Returns
-------
dict with keys: a, b, rho, m, sigma, rmse, converged, n_iter, arb_check
  arb_check: {butterfly_ok, min_variance_ok, butterfly_slack, min_variance}
)doc");

    // ── SVISurface class ─────────────────────────────────────────────────────
    py::class_<PySVISurface>(m, "SVISurface",
        R"doc(Multi-expiry SVI volatility surface with no-arbitrage checks.

Example
-------
>>> surf = vc.SVISurface()
>>> surf.add_slice(T=0.25, a=0.04, b=0.10, rho=-0.3, m=0.0, sigma=0.15)
>>> surf.add_slice(T=0.50, a=0.06, b=0.12, rho=-0.3, m=0.0, sigma=0.15)
>>> surf.implied_vol(K=0.0, T=0.25)    # log-moneyness k
>>> surf.is_arbitrage_free()
)doc")
        .def(py::init<>())
        .def("add_slice", &PySVISurface::add_slice,
             py::arg("T"), py::arg("a"), py::arg("b"),
             py::arg("rho"), py::arg("m"), py::arg("sigma"),
             "Add an expiry slice. Slices are automatically sorted by T.")
        .def("implied_vol", &PySVISurface::implied_vol,
             py::arg("k"), py::arg("T"),
             "Interpolated implied vol at log-moneyness k and expiry T.")
        .def("check_calendar", &PySVISurface::check_calendar,
             "Check calendar spread no-arb. Returns list of (T, is_ok) tuples.")
        .def("is_arbitrage_free", &PySVISurface::is_arbitrage_free,
             "True if all slices pass butterfly + calendar no-arb checks.");

    // ── Version ──────────────────────────────────────────────────────────────
    m.attr("__version__") = "0.1.0";
}
