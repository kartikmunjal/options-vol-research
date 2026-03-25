/**
 * bench_surface.cpp  —  Google Benchmark microbenchmarks for vol_core
 *
 * Benchmarks:
 *  BM_BSPrice          — Single BS call price
 *  BM_AllGreeks        — All 8 Greeks in one pass
 *  BM_AllGreeksSeparate — 8 individual Greek calls (baseline comparison)
 *  BM_ImpliedVol       — Single IV solve
 *  BM_IVStrip_N        — IV strip over 41/201 strikes (serial + parallel)
 *  BM_SVICalibrate     — SVI calibration on 11-strike smile
 *  BM_SVISurface       — 5-expiry surface build + vol query
 *
 * Build:
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
 *   cmake --build build
 *   ./build/bench_surface --benchmark_format=table
 */

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>
#include <numeric>

#include "../src/black_scholes.hpp"
#include "../src/iv_solver.hpp"
#include "../src/svi.hpp"

using namespace volcpp;

// ─────────────────────────────────────────────────────────────────────────────
// Common parameters
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double S   = 100.0;
static constexpr double K   = 100.0;
static constexpr double T   = 1.0;
static constexpr double r   = 0.05;
static constexpr double q   = 0.02;
static constexpr double sig = 0.20;

// ─────────────────────────────────────────────────────────────────────────────
// Black-Scholes
// ─────────────────────────────────────────────────────────────────────────────
static void BM_BSPrice(benchmark::State& state) {
    for (auto _ : state) {
        double p = BlackScholes<>::price(S, K, T, r, sig, true, q);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_BSPrice);

static void BM_AllGreeks(benchmark::State& state) {
    for (auto _ : state) {
        auto g = BlackScholes<>::all_greeks(S, K, T, r, sig, true, q);
        benchmark::DoNotOptimize(g);
    }
}
BENCHMARK(BM_AllGreeks)->Name("BM_AllGreeks_singlePass");

static void BM_AllGreeksSeparate(benchmark::State& state) {
    for (auto _ : state) {
        double p  = BlackScholes<>::price(S, K, T, r, sig, true,  q);
        double d  = BlackScholes<>::delta(S, K, T, r, sig, true,  q);
        double g  = BlackScholes<>::gamma(S, K, T, r, sig, q);
        double th = BlackScholes<>::theta(S, K, T, r, sig, true,  q);
        double v  = BlackScholes<>::vega (S, K, T, r, sig, q);
        double vn = BlackScholes<>::vanna(S, K, T, r, sig, q);
        double vg = BlackScholes<>::volga(S, K, T, r, sig, q);
        double ch = BlackScholes<>::charm(S, K, T, r, sig, true,  q);
        double rh = BlackScholes<>::rho  (S, K, T, r, sig, true,  q);
        benchmark::DoNotOptimize(p + d + g + th + v + vn + vg + ch + rh);
    }
}
BENCHMARK(BM_AllGreeksSeparate)->Name("BM_AllGreeks_8separate");

// ─────────────────────────────────────────────────────────────────────────────
// IV Solver
// ─────────────────────────────────────────────────────────────────────────────
static void BM_ImpliedVol(benchmark::State& state) {
    double market_price = BlackScholes<>::price(S, K, T, r, sig, true, q);
    for (auto _ : state) {
        double iv = IVSolver<>::solve(market_price, S, K, T, r, true, q);
        benchmark::DoNotOptimize(iv);
    }
}
BENCHMARK(BM_ImpliedVol);

// Strip: N strikes for a single expiry
static void BM_IVStrip(benchmark::State& state) {
    int N = state.range(0);
    std::vector<double> strikes(N), prices(N);
    for (int i = 0; i < N; ++i) {
        strikes[i] = 70.0 + (60.0 / (N - 1)) * i;  // 70 to 130
        prices[i]  = BlackScholes<>::price(S, strikes[i], T, r, sig, true, 0.0);
    }

    for (auto _ : state) {
        auto ivs = IVSolver<>::solve_strip(prices, strikes, S, T, r, true, 0.0);
        benchmark::DoNotOptimize(ivs.data());
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_IVStrip)->Range(8, 512)->Name("BM_IVStrip_N");

// ─────────────────────────────────────────────────────────────────────────────
// SVI calibration
// ─────────────────────────────────────────────────────────────────────────────
static void BM_SVICalibrate(benchmark::State& state) {
    int N = state.range(0);
    SVIParams truth{0.04, 0.10, -0.30, 0.0, 0.15};
    double T_svi = 0.5;

    std::vector<double> k_vec(N), iv_vec(N);
    for (int i = 0; i < N; ++i) {
        k_vec[i] = -0.3 + (0.6 / (N - 1)) * i;
        iv_vec[i] = SVI::implied_vol(k_vec[i], T_svi, truth);
    }

    for (auto _ : state) {
        auto result = SVI::calibrate(k_vec, iv_vec, T_svi, 4);
        benchmark::DoNotOptimize(result.rmse);
    }
}
BENCHMARK(BM_SVICalibrate)->Arg(11)->Arg(21)->Arg(41)->Name("BM_SVICalibrate_N");

// ─────────────────────────────────────────────────────────────────────────────
// Multi-expiry surface construction + vol query
// ─────────────────────────────────────────────────────────────────────────────
static void BM_SVISurface(benchmark::State& state) {
    int n_expiries = state.range(0);
    int n_strikes  = 21;
    double T_max   = 2.0;

    // Build surface data
    std::vector<double> expiries(n_expiries);
    for (int i = 0; i < n_expiries; ++i)
        expiries[i] = 0.25 + (T_max / n_expiries) * i;

    SVIParams truth{0.04, 0.10, -0.30, 0.0, 0.15};

    for (auto _ : state) {
        SVISurface surf;
        for (double Ts : expiries) {
            std::vector<double> k_vec(n_strikes), iv_vec(n_strikes);
            for (int j = 0; j < n_strikes; ++j) {
                k_vec[j]  = -0.3 + (0.6 / (n_strikes - 1)) * j;
                iv_vec[j] = SVI::implied_vol(k_vec[j], Ts, truth);
            }
            auto res = SVI::calibrate(k_vec, iv_vec, Ts);
            SVISlice sl;
            sl.T      = Ts;
            sl.params = res.params;
            surf.add_slice(std::move(sl));
        }
        // Query surface at several points
        double total = 0.0;
        for (double kq = -0.2; kq <= 0.2; kq += 0.1)
            for (double Tq : expiries)
                total += surf.implied_vol(kq, Tq);
        benchmark::DoNotOptimize(total);
    }
}
BENCHMARK(BM_SVISurface)->Arg(3)->Arg(5)->Arg(10)->Name("BM_SVISurface_expiries");

// ─────────────────────────────────────────────────────────────────────────────
// Surface pricing: evaluate BS price over a 2D (K, T) grid
// ─────────────────────────────────────────────────────────────────────────────
static void BM_PriceGrid(benchmark::State& state) {
    int N = state.range(0);  // N x N grid
    std::vector<double> strikes(N), expiries(N);
    for (int i = 0; i < N; ++i) {
        strikes[i]  = 70.0 + (60.0 / (N - 1)) * i;
        expiries[i] = 0.1  + (1.9  / (N - 1)) * i;
    }

    for (auto _ : state) {
        double total = 0.0;
        for (double Ti : expiries)
            for (double Ki : strikes)
                total += BlackScholes<>::price(S, Ki, Ti, r, sig, true, q);
        benchmark::DoNotOptimize(total);
    }
    state.SetItemsProcessed(state.iterations() * (long long)N * N);
}
BENCHMARK(BM_PriceGrid)->Range(10, 200)->Name("BM_PriceGrid_NxN");

// ─────────────────────────────────────────────────────────────────────────────
// Run benchmarks
// ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_MAIN();
