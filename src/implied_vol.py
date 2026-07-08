import numpy as np
from scipy.optimize import brentq

from black_scholes import black_scholes_price


def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type,
    q=0.0,
    sigma_lower=1e-6,
    sigma_upper=5.0,
    tolerance=1e-6,
):
    """
    Solve for implied volatility using Brent's method.

    Returns volatility as a decimal.
    Example: 0.20 means 20% annualized volatility.
    """

    if market_price <= 0:
        return np.nan

    if S <= 0 or K <= 0 or T <= 0:
        return np.nan

    intrinsic_value = max(S - K, 0) if option_type == "C" else max(K - S, 0)

    if market_price < intrinsic_value:
        return np.nan

    def objective_function(sigma):
        model_price = black_scholes_price(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            q=q,
        )
        return float(model_price - market_price)

    try:
        return brentq(
            objective_function,
            sigma_lower,
            sigma_upper,
            xtol=tolerance,
        )
    except ValueError:
        return np.nan