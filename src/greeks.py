import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    d1 = (
        np.log(S / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2


def delta(S, K, T, r, sigma, option_type, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    option_type = np.asarray(option_type)

    call_delta = np.exp(-q * T) * norm.cdf(d1)
    put_delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

    return np.where(option_type == "C", call_delta, put_delta)


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)

    return (
        np.exp(-q * T)
        * norm.pdf(d1)
        / (S * sigma * np.sqrt(T))
    )


def vega(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)

    return (
        S
        * np.exp(-q * T)
        * norm.pdf(d1)
        * np.sqrt(T)
        / 100
    )


def theta(S, K, T, r, sigma, option_type, q=0.0):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    option_type = np.asarray(option_type)

    first_term = (
        -S
        * np.exp(-q * T)
        * norm.pdf(d1)
        * sigma
        / (2 * np.sqrt(T))
    )

    call_theta = (
        first_term
        - r * K * np.exp(-r * T) * norm.cdf(d2)
        + q * S * np.exp(-q * T) * norm.cdf(d1)
    ) / 365

    put_theta = (
        first_term
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
        - q * S * np.exp(-q * T) * norm.cdf(-d1)
    ) / 365

    return np.where(option_type == "C", call_theta, put_theta)