/**
 * svi.hpp — Stochastic Volatility Inspired (SVI) surface
 *
 * Gatheral's SVI parameterisation (2004)
 * ---------------------------------------
 * The total implied variance as a function of log-moneyness k = ln(K/F):
 *
 *   w(k; θ) = a + b · [ρ(k − m) + √((k − m)² + σ²)]
 *
 * where  w = σ_imp² · T,  and:
 *   a  ∈ (−∞, 1)         vertical translation (overall variance level)
 *   b  ∈ [0, 4/T]         slope / angle (b ≥ 0 guarantees w bounded below)
 *   ρ  ∈ (−1, 1)         correlation / asymmetry  (ρ < 0 → left skew)
 *   m  ∈ ℝ               ATM horizontal shift
 *   σ  ∈ (0, ∞)           curvature / smile width
 *
 * Arbitrage-free conditions (Gatheral 2004)
 * ------------------------------------------
 * 1. Butterfly (Lee's moment formula):
 *        b · (1 + |ρ|) ≤ 4/T
 *    Violation ⟹ negative local variance / density ⟹ static butterfly arb.
 *    This is the only non-trivial constraint; min-variance is automatic
 *    when b ≥ 0 and σ > 0.
 *
 * 2. Calendar spread: total variance must be non-decreasing across expiries.
 *        w(k, T₂) ≥ w(k, T₁)  ∀k  when T₂ > T₁
 *    Checked slice-by-slice after multi-expiry calibration.
 *
 * Calibration
 * -----------
 * Minimises weighted MSE on total variance:
 *
 *   f(θ) = Σᵢ wᵢ [w(kᵢ; θ) − ŵᵢ]²  +  λ · max(0, b(1+|ρ|) − 4/T)²
 *
 * where ŵᵢ = σ̂_imp,i² · T.  Working in total-variance space is standard:
 * it makes the residual scale-invariant and avoids the non-linearity of
 * taking square roots.
 *
 * The analytic gradient ∇f(θ) is derived below and used in a projected
 * gradient descent with Armijo backtracking line search.  Four restarts from
 * diversified initial points guard against local optima.
 *
 * References
 * ----------
 *   Gatheral (2004) "A parsimonious arbitrage-free implied volatility
 *       parameterization with application to the valuation of volatility
 *       derivatives"  Merrill Lynch presentation.
 *   Gatheral & Jacquier (2014) "Arbitrage-free SVI volatility surfaces"
 *       Quantitative Finance 14(1).
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace volcpp {

// ─── Parameter struct ────────────────────────────────────────────────────────

struct SVIParams {
    double a;      ///< vertical translation
    double b;      ///< slope / wing angle  (b ≥ 0)
    double rho;    ///< asymmetry / skew    (|ρ| < 1)
    double m;      ///< ATM shift
    double sigma;  ///< smile curvature     (σ > 0)

    [[nodiscard]] bool is_valid() const noexcept {
        return b >= 0.0 && std::abs(rho) < 1.0 && sigma > 0.0;
    }
};

// ─── Arbitrage diagnostic ────────────────────────────────────────────────────

struct ArbitrageCheck {
    bool   butterfly_ok;     ///< b(1+|ρ|) ≤ 4/T
    bool   min_variance_ok;  ///< min w(k) ≥ 0 on a dense grid
    bool   calendar_ok;      ///< set to false by surface-level check if violated
    double butterfly_slack;  ///< 4/T − b(1+|ρ|) ; positive ⟹ no arb
    double min_variance;     ///< smallest w(k) observed on the check grid
};

// ─── Calibration result ──────────────────────────────────────────────────────

struct SVICalibResult {
    SVIParams      params;
    double         rmse;        ///< RMSE on implied vol (not total variance)
    bool           converged;
    int            n_iter;
    ArbitrageCheck arb_check;
};

// ─── Multi-expiry surface slice ──────────────────────────────────────────────

struct SVISlice {
    double         T;        ///< expiry in years
    double         forward;  ///< forward price F = S e^{(r−q)T}
    SVICalibResult calib;
};

// ─── Core SVI class ──────────────────────────────────────────────────────────

class SVI {
public:
    // ── Total variance  w(k; θ) ──────────────────────────────────────────────

    [[nodiscard]] static double total_variance(double k, const SVIParams& p) noexcept {
        double km  = k - p.m;
        double sqr = std::sqrt(km * km + p.sigma * p.sigma);
        return p.a + p.b * (p.rho * km + sqr);
    }

    // ── Implied vol  σ_imp(k, T; θ) ──────────────────────────────────────────

    [[nodiscard]] static double implied_vol(double k, double T, const SVIParams& p) noexcept {
        double w = total_variance(k, p);
        if (w <= 0.0) return std::numeric_limits<double>::quiet_NaN();
        return std::sqrt(w / T);
    }

    // ── Analytic gradient  ∇_θ w(k; θ) ──────────────────────────────────────
    // Returns [∂w/∂a, ∂w/∂b, ∂w/∂ρ, ∂w/∂m, ∂w/∂σ]

    [[nodiscard]] static std::array<double, 5>
    grad_w(double k, const SVIParams& p) noexcept {
        double km     = k - p.m;
        double sqr    = std::sqrt(km * km + p.sigma * p.sigma);
        double inv    = 1.0 / sqr;  // 1 / √(·)
        return {
            1.0,                          // ∂w/∂a
            p.rho * km + sqr,             // ∂w/∂b
            p.b * km,                     // ∂w/∂ρ
            -p.b * (p.rho + km * inv),    // ∂w/∂m
            p.b * p.sigma * inv           // ∂w/∂σ
        };
    }

    // ── Butterfly arbitrage check ────────────────────────────────────────────
    // Necessary condition (Lee 2004): b · (1 + |ρ|) ≤ 4/T

    [[nodiscard]] static bool butterfly_free(const SVIParams& p, double T) noexcept {
        return p.b * (1.0 + std::abs(p.rho)) <= 4.0 / T + 1e-10;
    }

    // ── Full arbitrage diagnostic ────────────────────────────────────────────

    [[nodiscard]] static ArbitrageCheck check_arbitrage(
        const SVIParams& p, double T,
        double k_lo = -2.5, double k_hi = 2.5,
        int n_grid = 500)
    {
        ArbitrageCheck ac;
        ac.butterfly_slack = 4.0 / T - p.b * (1.0 + std::abs(p.rho));
        ac.butterfly_ok    = ac.butterfly_slack >= -1e-10;
        ac.calendar_ok     = true;  // updated by SVISurface::check_calendar()

        double min_w = std::numeric_limits<double>::max();
        for (int i = 0; i <= n_grid; ++i) {
            double k = k_lo + (k_hi - k_lo) * double(i) / n_grid;
            min_w = std::min(min_w, total_variance(k, p));
        }
        ac.min_variance    = min_w;
        ac.min_variance_ok = min_w >= -1e-10;
        return ac;
    }

    // ── Calibration ──────────────────────────────────────────────────────────
    //
    // Fits the SVI parameter vector θ = (a, b, ρ, m, σ) to a set of
    // (log-moneyness, market-IV) pairs for a single expiry T.
    //
    // Uses projected gradient descent with:
    //   - Analytic gradient (avoids finite differences)
    //   - Armijo backtracking line search (robust step size)
    //   - Box-projection after each step (enforces bounds)
    //   - Butterfly penalty term (soft constraint)
    //   - 4 diverse starting points (guards against local optima)

    [[nodiscard]] static SVICalibResult calibrate(
        const std::vector<double>& k_vec,
        const std::vector<double>& market_iv,
        double T,
        const std::vector<double>& weights = {})
    {
        if (k_vec.size() < 3)
            throw std::invalid_argument("SVI calibration requires at least 3 strikes");
        if (k_vec.size() != market_iv.size())
            throw std::invalid_argument("k_vec and market_iv must have the same length");

        // Convert IV → total variance  ŵ = σ̂² · T
        std::vector<double> w_market(k_vec.size());
        for (std::size_t i = 0; i < k_vec.size(); ++i)
            w_market[i] = market_iv[i] * market_iv[i] * T;

        // Compute a representative ATM variance for initial guesses
        double atm_var = 0.0;
        for (double w : w_market) atm_var += w;
        atm_var = std::max(atm_var / w_market.size(), 0.001);

        // Parameter bounds
        const double b_max = std::min(4.0 / T, 2.0);
        Bounds bnd{-1.0, 1.0, 0.0, b_max, -0.999, 0.999, -1.0, 1.0, 1e-4, 2.0};

        // Four diverse starting points (equity-smile-motivated)
        const std::vector<SVIParams> inits = {
            {atm_var,       0.10, -0.50,  0.00, 0.10},  // baseline
            {atm_var * 0.8, 0.20, -0.70, -0.10, 0.15},  // typical equity skew
            {atm_var,       0.05, -0.30,  0.00, 0.30},  // low skew, wide smile
            {atm_var * 0.5, 0.30, -0.90, -0.20, 0.05},  // steep skew, narrow
        };

        SVICalibResult best;
        best.rmse = std::numeric_limits<double>::max();

        for (const auto& p0 : inits) {
            SVIParams p = clamp_to_bounds(p0, bnd);
            auto res = gradient_descent(p, T, k_vec, w_market, weights, bnd);
            if (res.rmse < best.rmse) best = res;
        }

        best.arb_check = check_arbitrage(best.params, T);
        return best;
    }

private:
    // ── Parameter bounds (a, b, rho, m, sigma) ───────────────────────────────
    struct Bounds {
        double a_lo, a_hi;
        double b_lo, b_hi;
        double r_lo, r_hi;
        double m_lo, m_hi;
        double s_lo, s_hi;
    };

    static SVIParams clamp_to_bounds(SVIParams p, const Bounds& bnd) noexcept {
        p.a     = std::clamp(p.a,     bnd.a_lo, bnd.a_hi);
        p.b     = std::clamp(p.b,     bnd.b_lo, bnd.b_hi);
        p.rho   = std::clamp(p.rho,   bnd.r_lo, bnd.r_hi);
        p.m     = std::clamp(p.m,     bnd.m_lo, bnd.m_hi);
        p.sigma = std::clamp(p.sigma, bnd.s_lo, bnd.s_hi);
        return p;
    }

    // ── Objective + analytic gradient ────────────────────────────────────────
    // Returns {f, ∇f} where f = Σ wᵢ(ŵᵢ − w_fit)² + butterfly_penalty

    struct ObjGrad { double value; std::array<double, 5> grad; };

    static ObjGrad objective_and_grad(
        const SVIParams& p, double T,
        const std::vector<double>& k_vec,
        const std::vector<double>& w_market,
        const std::vector<double>& weights)
    {
        double obj = 0.0;
        std::array<double, 5> g = {};

        for (std::size_t i = 0; i < k_vec.size(); ++i) {
            double w_fit = total_variance(k_vec[i], p);
            double res   = w_fit - w_market[i];
            double wi    = weights.empty() ? 1.0 : weights[i];
            obj += wi * res * res;
            auto dw = grad_w(k_vec[i], p);
            for (int j = 0; j < 5; ++j)
                g[j] += 2.0 * wi * res * dw[j];
        }

        // Soft butterfly penalty: λ · max(0, b(1+|ρ|) − 4/T)²
        constexpr double lambda = 1e5;
        double slack = 4.0 / T - p.b * (1.0 + std::abs(p.rho));
        if (slack < 0.0) {
            double pen_coeff = 2.0 * lambda * slack;  // 2λ · (slack)  (slack < 0)
            obj  += lambda * slack * slack;
            g[1] -= pen_coeff * (1.0 + std::abs(p.rho));              // ∂/∂b
            g[2] -= pen_coeff * p.b * (p.rho >= 0.0 ? 1.0 : -1.0);   // ∂/∂ρ
        }

        return {obj, g};
    }

    // ── Projected gradient descent with Armijo backtracking ──────────────────

    static SVICalibResult gradient_descent(
        SVIParams p, double T,
        const std::vector<double>& k_vec,
        const std::vector<double>& w_market,
        const std::vector<double>& weights,
        const Bounds& bnd,
        int    max_iter = 3000,
        double tol_obj  = 1e-12)
    {
        auto to_arr = [](const SVIParams& q) -> std::array<double, 5> {
            return {q.a, q.b, q.rho, q.m, q.sigma};
        };
        auto from_arr = [](const std::array<double, 5>& x) -> SVIParams {
            return {x[0], x[1], x[2], x[3], x[4]};
        };
        auto project = [&](std::array<double, 5>& x) {
            x[0] = std::clamp(x[0], bnd.a_lo, bnd.a_hi);
            x[1] = std::clamp(x[1], bnd.b_lo, bnd.b_hi);
            x[2] = std::clamp(x[2], bnd.r_lo, bnd.r_hi);
            x[3] = std::clamp(x[3], bnd.m_lo, bnd.m_hi);
            x[4] = std::clamp(x[4], bnd.s_lo, bnd.s_hi);
        };

        auto x   = to_arr(p);
        double prev_obj = std::numeric_limits<double>::max();
        int n_iter = 0;

        for (int iter = 0; iter < max_iter; ++iter) {
            auto [obj, grad] = objective_and_grad(from_arr(x), T, k_vec, w_market, weights);

            if (std::abs(obj - prev_obj) < tol_obj) { n_iter = iter; break; }
            prev_obj = obj;
            n_iter   = iter;

            // ‖∇f‖² for Armijo sufficient-decrease criterion
            double gnorm_sq = 0.0;
            for (double gi : grad) gnorm_sq += gi * gi;
            if (gnorm_sq < tol_obj * tol_obj) break;

            // Armijo backtracking: find step α s.t. f(x − α g) ≤ f(x) − c₁ α ‖g‖²
            constexpr double c1 = 1e-4;
            double alpha = 1e-3;
            for (int ls = 0; ls < 40; ++ls) {
                auto x_new = x;
                for (int j = 0; j < 5; ++j) x_new[j] -= alpha * grad[j];
                project(x_new);
                auto [obj_new, _] = objective_and_grad(from_arr(x_new), T,
                                                        k_vec, w_market, weights);
                if (obj_new <= obj - c1 * alpha * gnorm_sq) {
                    x = x_new;
                    alpha = std::min(alpha * 1.2, 1.0);  // grow step for next iter
                    break;
                }
                alpha *= 0.5;
                if (ls == 39) { x = x_new; }  // accept tiny step rather than stall
            }
        }

        p = from_arr(x);
        SVICalibResult res;
        res.params    = p;
        res.rmse      = compute_iv_rmse(p, T, k_vec, w_market);
        res.converged = n_iter < max_iter - 1;
        res.n_iter    = n_iter;
        return res;
    }

    // ── RMSE on implied vol ───────────────────────────────────────────────────

    static double compute_iv_rmse(
        const SVIParams& p, double T,
        const std::vector<double>& k_vec,
        const std::vector<double>& w_market)
    {
        double mse = 0.0;
        for (std::size_t i = 0; i < k_vec.size(); ++i) {
            double w_fit = std::max(total_variance(k_vec[i], p), 0.0);
            double iv_fit = std::sqrt(w_fit / T);
            double iv_mkt = std::sqrt(w_market[i] / T);
            double res    = iv_fit - iv_mkt;
            mse += res * res;
        }
        return std::sqrt(mse / k_vec.size());
    }
};

// ─── Multi-expiry SVI surface ─────────────────────────────────────────────────

class SVISurface {
public:
    std::vector<SVISlice> slices;  ///< sorted by ascending T

    void add_slice(SVISlice slice) {
        slices.push_back(std::move(slice));
        std::sort(slices.begin(), slices.end(),
                  [](const SVISlice& a, const SVISlice& b) { return a.T < b.T; });
    }

    // Query σ_imp at (strike K, expiry T) using nearest-expiry interpolation.
    // For production use, replace with term-structure interpolation in
    // total-variance space (ensures calendar-arb-free surface).
    [[nodiscard]] double implied_vol(double K, double T) const {
        if (slices.empty())
            throw std::runtime_error("SVISurface: no slices calibrated");
        const SVISlice* nearest = &slices.front();
        double min_dist = std::abs(T - nearest->T);
        for (const auto& s : slices) {
            double d = std::abs(T - s.T);
            if (d < min_dist) { min_dist = d; nearest = &s; }
        }
        double k = std::log(K / nearest->forward);
        return SVI::implied_vol(k, nearest->T, nearest->calib.params);
    }

    // Calendar-spread arbitrage check across all adjacent slice pairs.
    // Returns {expiry, is_calendar_free} for each pair (T₁, T₂).
    [[nodiscard]] std::vector<std::pair<double, bool>> check_calendar() const {
        std::vector<std::pair<double, bool>> results;
        constexpr int N_GRID = 200;
        for (std::size_t i = 1; i < slices.size(); ++i) {
            bool ok = true;
            for (int j = -N_GRID; j <= N_GRID; ++j) {
                double k  = j * 0.025;  // k ∈ [−5, 5] in steps of 0.025
                double w1 = SVI::total_variance(k, slices[i-1].calib.params);
                double w2 = SVI::total_variance(k, slices[i].calib.params);
                if (w2 < w1 - 1e-8) { ok = false; break; }
            }
            // Update the slice's calendar flag in-place
            const_cast<SVISlice&>(slices[i]).calib.arb_check.calendar_ok = ok;
            results.push_back({slices[i].T, ok});
        }
        return results;
    }

    [[nodiscard]] bool is_fully_arbitrage_free() const {
        for (const auto& s : slices) {
            if (!s.calib.arb_check.butterfly_ok)   return false;
            if (!s.calib.arb_check.min_variance_ok) return false;
        }
        for (const auto& [T, ok] : check_calendar())
            if (!ok) return false;
        return true;
    }
};

} // namespace volcpp
