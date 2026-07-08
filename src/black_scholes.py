import numpy as np
from scipy.stats import norm


def black_scholes_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type,
    q=0.0,
):
    """
    Price European call and put options using the Black-Scholes-Merton model.

    Parameters
    ----------
    S : float or array-like
        Current underlying price.
    K : float or array-like
        Strike price.
    T : float or array-like
        Time to expiration in years.
    r : float or array-like
        Risk-free interest rate.
    sigma : float or array-like
        Volatility as a decimal.
    option_type : str or array-like
        'C' for call, 'P' for put.
    q : float or array-like
        Continuous dividend yield. Default is 0.0.

    Returns
    -------
    float or np.ndarray
        Theoretical option price.
    """

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    option_type = np.asarray(option_type)

    if np.any(S <= 0):
        raise ValueError("Underlying price S must be positive.")
    if np.any(K <= 0):
        raise ValueError("Strike K must be positive.")
    if np.any(T <= 0):
        raise ValueError("Time to expiration T must be positive.")
    if np.any(sigma <= 0):
        raise ValueError("Volatility sigma must be positive.")

    d1 = (
        np.log(S / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    call_price = (
        S * np.exp(-q * T) * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    put_price = (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * np.exp(-q * T) * norm.cdf(-d1)
    )

    return np.where(option_type == "C", call_price, put_price)