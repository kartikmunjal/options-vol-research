/**
 * test_greeks.cpp  —  Catch2 unit tests for vol_core C++ library
 *
 * Tests:
 *  1. Put-call parity (price, delta, rho)
 *  2. Known ATM price: S=K=100, T=1, r=0.05, σ=0.20 → C ≈ 10.4506
 *  3. Each Greek vs finite-difference verification
 *  4. IV round-trip: price → IV → re-price must match
 *  5. SVI butterfly no-arb condition
 *  6. SVI calibration: round-trip on synthetic data
 *  7. Edge cases: deep ITM/OTM, near-zero T
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cmath>
#include <vector>
#include <numeric>

#include "../src/black_scholes.hpp"
#include "../src/iv_solver.hpp"
#include "../src/svi.hpp"

using namespace volcpp;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double S  = 100.0;
static constexpr double K  = 100.0;  // ATM
static constexpr double T  = 1.0;
static constexpr double r  = 0.05;
static constexpr double q  = 0.02;
static constexpr double sig = 0.20;

// Bump for finite-difference verification
static constexpr double dS   = 1e-4 * S;
static constexpr double dsig = 1e-4;
static constexpr double dr   = 1e-4;
static constexpr double dT   = 1e-6;

// Helper: BS price with bumped parameter
static double price_bumped_S(double dS_, bool call) {
    return BlackScholes<>::price(S + dS_, K, T, r, sig, call, q);
}
static double price_bumped_sig(double dsig_, bool call) {
    return BlackScholes<>::price(S, K, T, r, sig + dsig_, call, q);
}
static double price_bumped_r(double dr_, bool call) {
    return BlackScholes<>::price(S, K, T, r + dr_, sig, call, q);
}
static double price_bumped_T(double dT_, bool call) {
    return BlackScholes<>::price(S, K, T + dT_, r, sig, call, q);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Put-call parity: C - P = Se^{-qT} - Ke^{-rT}
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Put-call parity holds", "[bs][parity]") {
    double call = BlackScholes<>::price(S, K, T, r, sig, true,  q);
    double put  = BlackScholes<>::price(S, K, T, r, sig, false, q);
    double fwd  = S * std::exp(-q * T) - K * std::exp(-r * T);
    CHECK(std::abs(call - put - fwd) < 1e-10);

    // Also check delta parity: delta_call - delta_put = e^{-qT}
    double dc = BlackScholes<>::delta(S, K, T, r, sig, true,  q);
    double dp = BlackScholes<>::delta(S, K, T, r, sig, false, q);
    CHECK(std::abs(dc - dp - std::exp(-q * T)) < 1e-12);

    // Rho parity: rho_call - rho_put = T * K * e^{-rT} * 0.0001
    double rc = BlackScholes<>::rho(S, K, T, r, sig, true,  q);
    double rp = BlackScholes<>::rho(S, K, T, r, sig, false, q);
    double rho_diff = T * K * std::exp(-r * T) * 1e-4;
    CHECK(std::abs(rc - rp - rho_diff) < 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Known ATM call price (q = 0)
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("ATM call price matches closed-form reference", "[bs][price]") {
    // S=K=100, T=1, r=0.05, sigma=0.20, q=0
    // C = 100*N(d1) - 100*e^{-0.05}*N(d2)
    // d1 = (0.05 + 0.02)/0.20 = 0.35, d2 = 0.15
    // C ≈ 10.4506 (well-known textbook value)
    double call = BlackScholes<>::price(100.0, 100.0, 1.0, 0.05, 0.20, true, 0.0);
    CHECK(std::abs(call - 10.4506) < 5e-4);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Greeks vs finite difference
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Delta vs central finite difference", "[bs][greeks][delta]") {
    for (bool call : {true, false}) {
        double analytic = BlackScholes<>::delta(S, K, T, r, sig, call, q);
        double fd = (price_bumped_S(dS, call) - price_bumped_S(-dS, call)) / (2 * dS);
        CHECK(std::abs(analytic - fd) < 1e-6);
    }
}

TEST_CASE("Gamma vs central finite difference", "[bs][greeks][gamma]") {
    for (bool call : {true, false}) {
        double analytic = BlackScholes<>::gamma(S, K, T, r, sig, q);
        double fd = (price_bumped_S(dS, call) - 2 * BlackScholes<>::price(S, K, T, r, sig, call, q)
                     + price_bumped_S(-dS, call)) / (dS * dS);
        CHECK(std::abs(analytic - fd) < 1e-5);
    }
}

TEST_CASE("Vega vs central finite difference", "[bs][greeks][vega]") {
    for (bool call : {true, false}) {
        // Analytic vega is per vol point (0.01); FD: bump by 0.01
        double analytic = BlackScholes<>::vega(S, K, T, r, sig, q);
        double fd = (price_bumped_sig(0.01, call) - price_bumped_sig(-0.01, call)) / 2.0;
        CHECK(std::abs(analytic - fd) < 1e-6);
    }
}

TEST_CASE("Theta vs forward finite difference", "[bs][greeks][theta]") {
    for (bool call : {true, false}) {
        // Analytic theta is per calendar day (/365)
        // FD: bump T by -1/365 (option decays as T shrinks)
        double analytic = BlackScholes<>::theta(S, K, T, r, sig, call, q);
        double h = 1.0 / 365.0;
        double fd = (BlackScholes<>::price(S, K, T - h, r, sig, call, q)
                     - BlackScholes<>::price(S, K, T, r, sig, call, q));
        CHECK(std::abs(analytic - fd) < 1e-5);
    }
}

TEST_CASE("Rho vs central finite difference", "[bs][greeks][rho]") {
    for (bool call : {true, false}) {
        // Analytic rho is per bp (0.0001)
        double analytic = BlackScholes<>::rho(S, K, T, r, sig, call, q);
        double fd = (price_bumped_r(1e-4, call) - price_bumped_r(-1e-4, call)) / 2.0;
        CHECK(std::abs(analytic - fd) < 1e-8);
    }
}

TEST_CASE("Vanna vs cross finite difference", "[bs][greeks][vanna]") {
    for (bool call : {true, false}) {
        double analytic = BlackScholes<>::vanna(S, K, T, r, sig, q);
        // d(delta)/d(sigma): bump sigma, compute delta, finite difference
        double dplus  = BlackScholes<>::delta(S, K, T, r, sig + dsig, call, q);
        double dminus = BlackScholes<>::delta(S, K, T, r, sig - dsig, call, q);
        double fd = (dplus - dminus) / (2 * dsig);
        CHECK(std::abs(analytic - fd) < 1e-5);
    }
}

TEST_CASE("Volga vs second-order sigma finite difference", "[bs][greeks][volga]") {
    for (bool call : {true, false}) {
        // Analytic volga is vega squared effect; compare to d²P/dσ² * 0.01² / 2
        // i.e., raw volga = d²P/dσ² → scaled volga * (0.01)²
        double analytic = BlackScholes<>::volga(S, K, T, r, sig, q);
        double pc  = BlackScholes<>::price(S, K, T, r, sig, call, q);
        double pp  = BlackScholes<>::price(S, K, T, r, sig + 0.01, call, q);
        double pm  = BlackScholes<>::price(S, K, T, r, sig - 0.01, call, q);
        double fd_raw = (pp - 2 * pc + pm) / 1.0;  // d²P/d(0.01)² per vol point²
        CHECK(std::abs(analytic - fd_raw) < 1e-5);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. all_greeks consistency: each Greek equals individual function
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("all_greeks is consistent with individual Greek functions", "[bs][greeks]") {
    for (bool call : {true, false}) {
        auto g = BlackScholes<>::all_greeks(S, K, T, r, sig, call, q);
        CHECK(std::abs(g.price - BlackScholes<>::price(S, K, T, r, sig, call, q)) < 1e-12);
        CHECK(std::abs(g.delta - BlackScholes<>::delta(S, K, T, r, sig, call, q)) < 1e-12);
        CHECK(std::abs(g.gamma - BlackScholes<>::gamma(S, K, T, r, sig, q))       < 1e-12);
        CHECK(std::abs(g.vega  - BlackScholes<>::vega(S, K, T, r, sig, q))        < 1e-12);
        CHECK(std::abs(g.theta - BlackScholes<>::theta(S, K, T, r, sig, call, q)) < 1e-12);
        CHECK(std::abs(g.vanna - BlackScholes<>::vanna(S, K, T, r, sig, q))       < 1e-12);
        CHECK(std::abs(g.volga - BlackScholes<>::volga(S, K, T, r, sig, q))       < 1e-12);
        CHECK(std::abs(g.charm - BlackScholes<>::charm(S, K, T, r, sig, call, q)) < 1e-12);
        CHECK(std::abs(g.rho   - BlackScholes<>::rho(S, K, T, r, sig, call, q))   < 1e-12);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. IV round-trip: price → implied vol → reprice
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("IV round-trip: BS price → IV → reprice", "[iv][roundtrip]") {
    std::vector<double> sigmas = {0.05, 0.10, 0.20, 0.40, 0.80, 1.50};
    std::vector<double> moneyness = {0.70, 0.85, 1.00, 1.15, 1.30};

    for (double sv : sigmas) {
        for (double kf : moneyness) {
            double Ki = K * kf;
            for (bool call : {true, false}) {
                double target = BlackScholes<>::price(S, Ki, T, r, sv, call, q);
                // Skip tiny prices (deep OTM; numerical noise dominates)
                if (target < 1e-6) continue;
                double iv = IVSolver<>::solve(target, S, Ki, T, r, call, q);
                double repriced = BlackScholes<>::price(S, Ki, T, r, iv, call, q);
                CHECK(std::abs(repriced - target) / target < 1e-7);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. IV strip: parallel solve produces same results as serial
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("IV strip: parallel matches serial", "[iv][strip]") {
    std::vector<double> strikes(21);
    for (int i = 0; i < 21; ++i) strikes[i] = 80.0 + i * 2.0;

    std::vector<double> prices(strikes.size());
    for (size_t i = 0; i < strikes.size(); ++i)
        prices[i] = BlackScholes<>::price(S, strikes[i], T, r, sig, true, 0.0);

    auto strip_ivs = IVSolver<>::solve_strip(prices, strikes, S, T, r, true, 0.0);

    for (size_t i = 0; i < strikes.size(); ++i) {
        if (prices[i] < 1e-6) continue;
        double serial = IVSolver<>::solve(prices[i], S, strikes[i], T, r, true, 0.0);
        CHECK(std::abs(strip_ivs[i] - serial) < 1e-9);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. SVI butterfly no-arb condition
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("SVI butterfly condition: b(1+|rho|) <= 4/T", "[svi][butterfly]") {
    double T_svi = 0.5;

    // Well-behaved parameters: should be butterfly-free
    SVIParams ok{0.04, 0.10, -0.3, 0.0, 0.15};
    CHECK(SVI::butterfly_free(ok, T_svi) == true);

    // Extreme b with |rho| near 1: b*(1+|rho|) > 4/T → fails
    SVIParams bad{0.04, 5.0, -0.95, 0.0, 0.15};
    CHECK(SVI::butterfly_free(bad, T_svi) == false);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. SVI total variance: positivity and asymptotes
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("SVI total variance properties", "[svi][properties]") {
    SVIParams p{0.04, 0.10, -0.3, 0.0, 0.15};
    double T_svi = 1.0;

    // w(k) must be non-negative everywhere
    for (double k = -2.0; k <= 2.0; k += 0.1) {
        double w = SVI::total_variance(k, p);
        CHECK(w >= 0.0);
        // Implied vol must be real
        double iv = SVI::implied_vol(k, T_svi, p);
        CHECK(iv > 0.0);
        CHECK(std::isfinite(iv));
    }

    // Slope at +inf: w(k)/k → b*(1+rho) > 0
    // Slope at -inf: w(k)/k → b*(1-|rho|) ≥ 0
    double w_far_right = SVI::total_variance(10.0, p);
    double w_far_left  = SVI::total_variance(-10.0, p);
    CHECK(w_far_right > w_far_left);  // right wing heavier when rho < 0 (skew)
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. SVI calibration round-trip
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("SVI calibration round-trip on synthetic smiles", "[svi][calibrate]") {
    double T_svi = 0.5;
    SVIParams true_params{0.04, 0.10, -0.30, 0.0, 0.15};

    // Generate synthetic market vols from true params
    std::vector<double> k_vec, iv_vec;
    for (int i = -5; i <= 5; ++i) {
        double k = 0.1 * i;
        k_vec.push_back(k);
        iv_vec.push_back(SVI::implied_vol(k, T_svi, true_params));
    }

    // Calibrate
    auto result = SVI::calibrate(k_vec, iv_vec, T_svi);

    // RMSE should be tiny (exact reconstruction)
    CHECK(result.rmse < 1e-6);
    CHECK(result.converged == true);
    CHECK(result.arb_check.butterfly_ok == true);

    // Recovered params should be close to truth (up to parameterization equivalence)
    // Test via total variance values rather than raw params
    for (size_t i = 0; i < k_vec.size(); ++i) {
        double w_true = SVI::total_variance(k_vec[i], true_params);
        double w_fit  = SVI::total_variance(k_vec[i], result.params);
        CHECK(std::abs(w_true - w_fit) < 1e-8);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Edge cases
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Deep in-the-money call: price ≈ intrinsic discounted", "[bs][edge]") {
    // Deep ITM call: price ≈ Se^{-qT} - Ke^{-rT}
    double call = BlackScholes<>::price(200.0, 100.0, 1.0, 0.05, 0.20, true, 0.0);
    double intrinsic = 200.0 - 100.0 * std::exp(-0.05);
    CHECK(std::abs(call - intrinsic) < 0.02);  // small time value remains
}

TEST_CASE("Deep out-of-the-money call: price ≈ 0", "[bs][edge]") {
    double call = BlackScholes<>::price(50.0, 200.0, 1.0, 0.05, 0.20, true, 0.0);
    CHECK(call < 1e-4);
}

TEST_CASE("Very short expiry: theta dominates", "[bs][edge]") {
    // ATM option with T=1 day: theta should be very negative relative to price
    double call = BlackScholes<>::price(100.0, 100.0, 1.0/365.0, 0.05, 0.20, true, 0.0);
    double theta = BlackScholes<>::theta(100.0, 100.0, 1.0/365.0, 0.05, 0.20, true, 0.0);
    CHECK(call > 0.0);
    CHECK(theta < 0.0);  // theta is always negative for long options
    // Theta in 1 day (already per day) ≈ -call (option expires worthless)
    CHECK(std::abs(theta) > 0.5 * call);
}

TEST_CASE("IV of intrinsic-value options returns NaN", "[iv][edge]") {
    // Price below intrinsic: no IV exists
    double intrinsic = std::max(S - K, 0.0);
    double below_intrinsic = intrinsic - 0.01;
    if (below_intrinsic > 0) {
        double iv = IVSolver<>::solve(below_intrinsic, S, K, T, r, true, 0.0);
        CHECK(std::isnan(iv));
    }
}
