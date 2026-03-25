/**
 * black_scholes.hpp — Templated Black-Scholes-Merton pricer + all eight
 *                     analytic Greeks.
 *
 * Design
 * ------
 * The central performance idea is that every Greek shares the same five
 * transcendental evaluations (log, sqrt, exp×2, erfc×2, exp for PDF).
 * `BSIntermediates` computes them once; every Greek reads from that cache
 * and runs in a handful of multiplications/additions.
 *
 * This makes `all_greeks()` essentially free: calling it costs the same as
 * calling `price()` alone plus a few FLOPs per Greek, rather than
 * recomputing d₁/d₂/Φ/φ from scratch for each.
 *
 * Greeks convention
 * -----------------
 *   delta  — ∂V/∂S
 *   gamma  — ∂²V/∂S²
 *   theta  — ∂V/∂t, annualised then divided by 365 → per calendar day
 *   vega   — ∂V/∂σ per 1 vol POINT (i.e. per 0.01 change in σ)
 *   vanna  — ∂²V/∂S∂σ  (also called DVegaDSpot / DDeltaDVol)
 *   volga  — ∂²V/∂σ²   (also called vomma)
 *   charm  — ∂Δ/∂t, per calendar day  (delta bleed)
 *   rho    — ∂V/∂r per 1 basis-point (0.01%) rate move
 *
 * References
 * ----------
 *   Black & Scholes (1973)  "The Pricing of Options and Corporate Liabilities"
 *   Merton (1973)           Continuous dividend yield extension
 */

#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace volcpp {

// ─── Normal distribution utilities ──────────────────────────────────────────
//
// ncdf uses the identity  Φ(x) = ½·erfc(−x/√2) which is accurate to
// double precision everywhere (unlike polynomial approximations).

template <typename T>
[[nodiscard]] inline T ncdf(T x) noexcept {
    return T(0.5) * std::erfc(-x * T(M_SQRT1_2));
}

template <typename T>
[[nodiscard]] inline T npdf(T x) noexcept {
    // 1/√(2π) to full double precision
    constexpr T INV_SQRT_2PI = T(0.3989422804014326779399460599);
    return INV_SQRT_2PI * std::exp(T(-0.5) * x * x);
}

// ─── Pre-computed intermediates (shared across all Greeks) ──────────────────

template <typename T>
struct BSIntermediates {
    T d1, d2;
    T sqrt_T, sigma_sqrt_T;
    T Nd1, Nd2;          //  Φ(+d₁),  Φ(+d₂)
    T Nnd1, Nnd2;        //  Φ(−d₁),  Φ(−d₂)  =  1 − Φ(d₁), 1 − Φ(d₂)
    T nd1;               //  φ(d₁)
    T Se_qT;             //  S · e^{−qT}
    T Ke_rT;             //  K · e^{−rT}
    T exp_qT;            //  e^{−qT}   (for delta, charm)
};

template <typename T>
[[nodiscard]] BSIntermediates<T> compute_intermediates(
    T S, T K, T T_exp, T r, T sigma, T q) noexcept
{
    BSIntermediates<T> im;
    im.sqrt_T        = std::sqrt(T_exp);
    im.sigma_sqrt_T  = sigma * im.sqrt_T;
    T log_moneyness  = std::log(S / K);
    im.d1  = (log_moneyness + (r - q + T(0.5) * sigma * sigma) * T_exp)
             / im.sigma_sqrt_T;
    im.d2  = im.d1 - im.sigma_sqrt_T;
    im.Nd1  = ncdf(im.d1);
    im.Nd2  = ncdf(im.d2);
    im.Nnd1 = T(1) - im.Nd1;
    im.Nnd2 = T(1) - im.Nd2;
    im.nd1  = npdf(im.d1);
    im.exp_qT = std::exp(-q * T_exp);
    im.Se_qT  = S * im.exp_qT;
    im.Ke_rT  = K * std::exp(-r * T_exp);
    return im;
}

// ─── Greeks result struct ────────────────────────────────────────────────────

template <typename T = double>
struct Greeks {
    T price;
    T delta;
    T gamma;
    T theta;  // per calendar day
    T vega;   // per vol point (0.01)
    T vanna;
    T volga;  // vomma
    T charm;  // per calendar day
    T rho;    // per basis point (0.0001)
};

// ─── BlackScholes — pricer + all analytic Greeks ────────────────────────────

template <typename T = double>
class BlackScholes {
    static_assert(std::is_floating_point_v<T>,
                  "BlackScholes<T> requires a floating-point type");
public:

    // ── Option price ─────────────────────────────────────────────────────────

    [[nodiscard]] static T price(
        T S, T K, T T_exp, T r, T sigma, bool is_call, T q = T(0))
    {
        if (T_exp <= T(0))
            return is_call ? std::max(S - K, T(0)) : std::max(K - S, T(0));
        return price_from(compute_intermediates(S, K, T_exp, r, sigma, q), is_call);
    }

    // Efficient overload: price from pre-computed intermediates
    [[nodiscard]] static T price_from(
        const BSIntermediates<T>& im, bool is_call) noexcept
    {
        return is_call
            ? im.Se_qT * im.Nd1 - im.Ke_rT * im.Nd2
            : im.Ke_rT * im.Nnd2 - im.Se_qT * im.Nnd1;
    }

    // ── Individual Greeks ─────────────────────────────────────────────────────
    // Each calls compute_intermediates internally.  When you need more than
    // one Greek, call all_greeks() instead — it is strictly cheaper.

    [[nodiscard]] static T delta(T S, T K, T T_exp, T r, T sigma, bool is_call, T q = T(0)) {
        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);
        return is_call ? im.exp_qT * im.Nd1 : im.exp_qT * (im.Nd1 - T(1));
    }

    [[nodiscard]] static T gamma(T S, T K, T T_exp, T r, T sigma, T q = T(0)) {
        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);
        return im.exp_qT * im.nd1 / (S * im.sigma_sqrt_T);
    }

    [[nodiscard]] static T vega(T S, T K, T T_exp, T r, T sigma, T q = T(0)) {
        // ∂V/∂σ per 1 vol point (0.01).  Raw vega = Se^{-qT} φ(d₁) √T.
        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);
        return im.Se_qT * im.nd1 * im.sqrt_T * T(0.01);
    }

    // Raw vega (dV/dσ without the 0.01 scaling) — used internally by IV solver
    [[nodiscard]] static T vega_raw(T S, T K, T T_exp, T r, T sigma, T q = T(0)) {
        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);
        return im.Se_qT * im.nd1 * im.sqrt_T;
    }

    [[nodiscard]] static T theta(T S, T K, T T_exp, T r, T sigma, bool is_call, T q = T(0)) {
        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);
        return theta_from(im, S, r, sigma, is_call, q);
    }

    [[nodiscard]] static T vanna(T S, T K, T T_exp, T r, T sigma, T q = T(0)) {
        // ∂²V / ∂S∂σ  =  −e^{−qT} φ(d₁) d₂ / σ
        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);
        return -im.exp_qT * im.nd1 * im.d2 / sigma;
    }

    [[nodiscard]] static T volga(T S, T K, T T_exp, T r, T sigma, T q = T(0)) {
        // ∂²V / ∂σ²  =  S e^{−qT} φ(d₁) √T  d₁ d₂ / σ
        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);
        return im.Se_qT * im.nd1 * im.sqrt_T * im.d1 * im.d2 / sigma;
    }

    [[nodiscard]] static T charm(T S, T K, T T_exp, T r, T sigma, bool is_call, T q = T(0)) {
        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);
        return charm_from(im, r, sigma, is_call, T_exp, q);
    }

    [[nodiscard]] static T rho(T S, T K, T T_exp, T r, T sigma, bool is_call, T q = T(0)) {
        // ∂V / ∂r  per 1 bp (0.0001)
        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);
        return is_call
            ?  T_exp * im.Ke_rT * im.Nd2  * T(0.0001)
            : -T_exp * im.Ke_rT * im.Nnd2 * T(0.0001);
    }

    // ── All eight Greeks from a single set of intermediates ──────────────────
    // Cost: 1×compute_intermediates + ~30 multiplications.
    // Equivalent to calling price() once — all remaining Greeks are negligible.

    [[nodiscard]] static Greeks<T> all_greeks(
        T S, T K, T T_exp, T r, T sigma, bool is_call, T q = T(0))
    {
        if (T_exp <= T(0)) {
            Greeks<T> g{};
            g.price = is_call ? std::max(S - K, T(0)) : std::max(K - S, T(0));
            // At expiry, delta is the Heaviside; all time Greeks vanish
            g.delta = is_call ? T(S > K) : -T(S < K);
            return g;
        }

        auto im = compute_intermediates(S, K, T_exp, r, sigma, q);

        Greeks<T> g;

        // ── Price ─────────────────────────────────────────────────────────
        g.price = price_from(im, is_call);

        // ── Delta  ∂V/∂S ──────────────────────────────────────────────────
        g.delta = is_call
            ? im.exp_qT * im.Nd1
            : im.exp_qT * (im.Nd1 - T(1));

        // ── Gamma  ∂²V/∂S² (identical for calls and puts) ────────────────
        g.gamma = im.exp_qT * im.nd1 / (S * im.sigma_sqrt_T);

        // ── Theta  ∂V/∂t per calendar day ────────────────────────────────
        g.theta = theta_from(im, S, r, sigma, is_call, q);

        // ── Vega  ∂V/∂σ per vol point ─────────────────────────────────────
        g.vega = im.Se_qT * im.nd1 * im.sqrt_T * T(0.01);

        // ── Vanna  ∂²V/∂S∂σ ──────────────────────────────────────────────
        g.vanna = -im.exp_qT * im.nd1 * im.d2 / sigma;

        // ── Volga (Vomma)  ∂²V/∂σ² ───────────────────────────────────────
        g.volga = im.Se_qT * im.nd1 * im.sqrt_T * im.d1 * im.d2 / sigma;

        // ── Charm  ∂Δ/∂t per calendar day ────────────────────────────────
        g.charm = charm_from(im, r, sigma, is_call, T_exp, q);

        // ── Rho  ∂V/∂r per bp ────────────────────────────────────────────
        g.rho = is_call
            ?  T_exp * im.Ke_rT * im.Nd2  * T(0.0001)
            : -T_exp * im.Ke_rT * im.Nnd2 * T(0.0001);

        return g;
    }

private:
    // Theta helper (reused by individual theta() and all_greeks())
    [[nodiscard]] static T theta_from(
        const BSIntermediates<T>& im, T S, T r, T sigma, bool is_call, T q) noexcept
    {
        // Annual theta = −(S e^{-qT} φ(d₁) σ) / (2√T) ∓ carry terms
        T common = -(im.Se_qT * im.nd1 * sigma) / (T(2) * im.sqrt_T);
        T annual;
        if (is_call)
            annual = common
                     - r * im.Ke_rT * im.Nd2
                     + q * im.Se_qT * im.Nd1;
        else
            annual = common
                     + r * im.Ke_rT * im.Nnd2
                     - q * im.Se_qT * im.Nnd1;
        return annual / T(365);  // per calendar day, matching Python convention
    }

    // Charm helper  ∂Δ/∂t = [δ · q − φ(d₁)/σ · (2(r−q)T − d₂ σ√T)/(2T)] / 365
    [[nodiscard]] static T charm_from(
        const BSIntermediates<T>& im, T r, T sigma, bool is_call,
        T T_exp, T q) noexcept
    {
        T numerator = T(2) * (r - q) * T_exp - im.d2 * im.sigma_sqrt_T;
        T charm_inner = im.exp_qT * im.nd1 * numerator
                        / (T(2) * T_exp * im.sigma_sqrt_T);
        T annual;
        if (is_call)
            annual =  q * im.exp_qT * im.Nd1  - charm_inner;
        else
            annual = -q * im.exp_qT * im.Nnd1 - charm_inner;
        return annual / T(365);
    }
};

// ─── Explicit instantiations ─────────────────────────────────────────────────
// (extern declarations; definitions live in bindings.cpp to avoid ODR issues)
extern template class BlackScholes<float>;
extern template class BlackScholes<double>;

} // namespace volcpp
