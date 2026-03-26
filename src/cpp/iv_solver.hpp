/**
 * iv_solver.hpp — Implied volatility solver
 *
 * Algorithm
 * ---------
 * Stage 1  Intrinsic-value guard: if the market price ≤ intrinsic (possibly
 *          under the forward discount) the option has no extrinsic value and
 *          IV is not defined — return NaN.
 *
 * Stage 2  Brenner-Subrahmanyam (1988) ATM initial guess:
 *              σ₀ ≈ √(2π/T) · (C / S)
 *          Accurate to ±5% for near-ATM options; quadratically fast start
 *          for Newton-Raphson near the money.
 *
 * Stage 3  Newton-Raphson using Vega as the Jacobian:
 *              σₙ₊₁ = σₙ − (BS(σₙ) − price) / ν(σₙ)
 *          where ν = S e^{-qT} φ(d₁) √T  (the raw, unscaled vega).
 *          We re-use the BSIntermediates from the price computation to avoid
 *          redundant transcendental evaluations per iteration.
 *          Exits when |error| < tol or vega → 0 (deep OTM) or step overshoots.
 *
 * Stage 4  Bisection fallback on [1e-4, 5.0]: guaranteed convergence in
 *          ≤ 100 iterations for monotone BS price.  Used for deep OTM/ITM
 *          options where NR stalls.
 *
 * Vectorised surface solve
 * -------------------------
 * `solve_strip()` applies the solver across a strip of strikes sharing the
 * same S, T, r, q.  If compiled with C++17 execution policies and TBB/pstl,
 * the transform runs in parallel (std::execution::par_unseq); otherwise it
 * falls back to the sequential path transparently.
 *
 * References
 * ----------
 *   Brenner & Subrahmanyam (1988) "A Simple Formula to Compute the Implied
 *       Standard Deviation"  Financial Analysts Journal.
 *   Manaster & Koehler (1982)  refinement of the initial guess.
 */

#pragma once

#include "black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Use parallel execution if available (C++17 + libpstl/TBB)
#if defined(__has_include) && __has_include(<execution>)
#include <execution>
#define VOLCPP_HAS_PARALLEL 1
#endif

namespace volcpp {

template <typename T = double>
class IVSolver {
    static_assert(std::is_floating_point_v<T>, "IVSolver<T> requires floating-point T");

public:
    // ── Solver knobs ─────────────────────────────────────────────────────────
    struct Options {
        T   tol          = T(1e-8);   // |BS_price − market_price| threshold
        int max_iter_nr  = 50;        // Newton-Raphson max iterations
        int max_iter_bis = 100;       // Bisection max iterations
        T   vol_lo       = T(1e-4);   // search bracket lower bound (0.01%)
        T   vol_hi       = T(5.0);    // search bracket upper bound (500%)
    };

    // ── Single-option solve ──────────────────────────────────────────────────

    [[nodiscard]] static T solve(
        T market_price, T S, T K, T T_exp, T r, bool is_call,
        T q = T(0), const Options& opts = {})
    {
        // Guard: T ≤ 0 → intrinsic, no IV
        if (T_exp <= T(0))
            return std::numeric_limits<T>::quiet_NaN();

        // Guard: market price ≤ theoretical intrinsic (arbitrage or bad data)
        const T disc    = std::exp(-r * T_exp);
        const T disc_q  = std::exp(-q * T_exp);
        const T intrinsic = is_call
            ? std::max(S * disc_q - K * disc, T(0))
            : std::max(K * disc - S * disc_q, T(0));
        if (market_price <= intrinsic + T(1e-10))
            return std::numeric_limits<T>::quiet_NaN();

        // Stage 2: Brenner-Subrahmanyam initial guess (ATM approximation)
        T sigma0 = std::sqrt(T(2) * T(M_PI) / T_exp) * (market_price / S);
        sigma0 = std::clamp(sigma0, opts.vol_lo, opts.vol_hi);

        // Stage 3: Newton-Raphson
        T sigma_nr = newton_raphson(market_price, S, K, T_exp, r, is_call, q, sigma0, opts);
        if (!std::isnan(sigma_nr)) return sigma_nr;

        // Stage 4: Bisection fallback
        return bisection(market_price, S, K, T_exp, r, is_call, q, opts);
    }

    // ── Vectorised strip solve (same S, T, r, q; varies K and market price) ─

    [[nodiscard]] static std::vector<T> solve_strip(
        const std::vector<T>& market_prices,
        const std::vector<T>& strikes,
        T S, T T_exp, T r, bool is_call,
        T q = T(0), const Options& opts = {})
    {
        std::vector<T> ivs(market_prices.size());

#ifdef VOLCPP_HAS_PARALLEL
        std::transform(
            std::execution::par_unseq,
            market_prices.begin(), market_prices.end(),
            strikes.begin(),
            ivs.begin(),
            [&](T price, T K) {
                return solve(price, S, K, T_exp, r, is_call, q, opts);
            }
        );
#else
        for (std::size_t i = 0; i < market_prices.size(); ++i)
            ivs[i] = solve(market_prices[i], S, strikes[i], T_exp, r, is_call, q, opts);
#endif
        return ivs;
    }

private:
    // Newton-Raphson: returns NaN if convergence fails (caller switches to bisection)
    [[nodiscard]] static T newton_raphson(
        T market_price, T S, T K, T T_exp, T r, bool is_call,
        T q, T sigma0, const Options& opts) noexcept
    {
        T sigma = sigma0;
        for (int i = 0; i < opts.max_iter_nr; ++i) {
            auto im = compute_intermediates(S, K, T_exp, r, sigma, q);

            // Reuse intermediates: price and raw vega from the same d₁/d₂
            T p   = BlackScholes<T>::price_from(im, is_call);
            T raw_vega = im.Se_qT * im.nd1 * im.sqrt_T;  // dPrice/dSigma
            T diff = p - market_price;

            // Convergence
            if (std::abs(diff) < opts.tol) return sigma;

            // Vega ≈ 0: deep OTM, NR can't proceed (would divide near-zero)
            if (std::abs(raw_vega) < T(1e-12))
                return std::numeric_limits<T>::quiet_NaN();

            T new_sigma = sigma - diff / raw_vega;

            // Overshot into non-physical territory
            if (new_sigma <= T(0) || new_sigma > opts.vol_hi)
                return std::numeric_limits<T>::quiet_NaN();

            sigma = new_sigma;
        }
        return std::numeric_limits<T>::quiet_NaN();
    }

    // Bisection on [vol_lo, vol_hi]: O(log(1/tol)) iterations, always converges
    [[nodiscard]] static T bisection(
        T market_price, T S, T K, T T_exp, T r, bool is_call,
        T q, const Options& opts) noexcept
    {
        T lo = opts.vol_lo, hi = opts.vol_hi;

        // Verify bracket (BS price is monotone increasing in σ)
        T p_lo = BlackScholes<T>::price(S, K, T_exp, r, lo, is_call, q);
        T p_hi = BlackScholes<T>::price(S, K, T_exp, r, hi, is_call, q);
        if (market_price < p_lo || market_price > p_hi)
            return std::numeric_limits<T>::quiet_NaN();

        for (int i = 0; i < opts.max_iter_bis; ++i) {
            T mid  = T(0.5) * (lo + hi);
            T p    = BlackScholes<T>::price(S, K, T_exp, r, mid, is_call, q);
            T diff = p - market_price;

            if (std::abs(diff) < opts.tol) return mid;
            if ((hi - lo) < opts.tol * T(1e-2)) return mid;

            if (diff < T(0)) lo = mid; else hi = mid;
        }
        return T(0.5) * (lo + hi);
    }
};

// ─── Explicit instantiations ─────────────────────────────────────────────────
extern template class IVSolver<float>;
extern template class IVSolver<double>;

} // namespace volcpp
