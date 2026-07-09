import numpy as np


def generate_quotes(
    fair_value,
    market_spread,
    implied_vol,
    inventory,
    net_delta,
    net_vega,
    base_spread=0.10,
    vol_spread_multiplier=0.50,
    liquidity_spread_multiplier=0.25,
    inventory_skew_multiplier=0.01,
    delta_skew_multiplier=0.0001,
    vega_skew_multiplier=0.00005,
):
    """
    Generate adaptive bid/ask quotes around theoretical fair value.

    Parameters
    ----------
    fair_value : float
        Theoretical option price.
    market_spread : float
        Observed market ask - bid.
    implied_vol : float
        Option implied volatility.
    inventory : float
        Current position in this option contract.
    net_delta : float
        Current portfolio delta exposure.
    net_vega : float
        Current portfolio vega exposure.
    base_spread : float
        Minimum spread quoted by our market maker.

    Returns
    -------
    dict
        bid, ask, quoted_spread, quote_skew
    """

    if fair_value <= 0 or np.isnan(fair_value):
        return {
            "bid": np.nan,
            "ask": np.nan,
            "quoted_spread": np.nan,
            "quote_skew": np.nan,
        }

    volatility_component = vol_spread_multiplier * implied_vol
    liquidity_component = liquidity_spread_multiplier * market_spread

    quoted_spread = base_spread + volatility_component + liquidity_component

    quote_skew = (
        inventory_skew_multiplier * inventory
        + delta_skew_multiplier * net_delta
        + vega_skew_multiplier * net_vega
    )

    bid = fair_value - quoted_spread / 2 - quote_skew
    ask = fair_value + quoted_spread / 2 - quote_skew

    bid = max(bid, 0.01)
    ask = max(ask, bid + 0.01)

    return {
        "bid": bid,
        "ask": ask,
        "quoted_spread": quoted_spread,
        "quote_skew": quote_skew,
    }