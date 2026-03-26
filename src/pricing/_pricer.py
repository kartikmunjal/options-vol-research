"""
Pricer dispatch layer — tries C++ vol_core first, falls back to Python.

The C++ extension uses ``is_call: bool`` while the original Python module uses
``option_type: str`` ('call'/'put').  This module normalises both and exposes
the same ``price``, ``delta``, ``all_greeks`` signatures the backtest uses, so
delta_hedge.py needs zero changes to its import line.

Priority:  vol_core (C++) > src/pricing/black_scholes.py (Python)
"""
from __future__ import annotations
import logging

__all__ = ["price", "delta", "all_greeks"]

log = logging.getLogger(__name__)

# ── Try C++ extension ─────────────────────────────────────────────────────────
try:
    import vol_core as _vc  # type: ignore[import]  # compiled C++ extension

    _BACKEND = "C++ (vol_core)"

    def price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call", q: float = 0.0) -> float:
        return float(_vc.bs_price(S, K, T, r, sigma, option_type == "call", q))

    def delta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call", q: float = 0.0) -> float:
        return float(_vc.bs_delta(S, K, T, r, sigma, option_type == "call", q))

    def all_greeks(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: str = "call", q: float = 0.0) -> dict:
        g = _vc.bs_all_greeks(S, K, T, r, sigma, option_type == "call", q)
        # C++ returns theta/charm per calendar day and vega per vol point — same
        # convention as the Python module, so no scaling needed.
        return {k: float(v) for k, v in g.items()}

# ── Fall back to Python ───────────────────────────────────────────────────────
except ImportError:
    _BACKEND = "Python (black_scholes.py)"
    from src.pricing.black_scholes import (  # type: ignore
        price, delta, all_greeks,
    )

log.debug("Pricer backend: %s", _BACKEND)
