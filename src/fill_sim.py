import numpy as np


def estimate_fill_probability(
    quoted_spread,
    market_spread,
    volume,
    base_fill_rate=0.20,
    spread_sensitivity=1.50,
    volume_sensitivity=0.10,
):
    """
    Estimate probability that our quote gets filled.

    Narrower quotes and higher-volume contracts should receive more fills.
    Wider quotes and lower-volume contracts should receive fewer fills.
    """

    if quoted_spread <= 0 or market_spread <= 0:
        return 0.0

    spread_ratio = quoted_spread / market_spread

    spread_effect = np.exp(-spread_sensitivity * (spread_ratio - 1))
    volume_effect = 1 + volume_sensitivity * np.log1p(volume)

    fill_probability = base_fill_rate * spread_effect * volume_effect

    return float(np.clip(fill_probability, 0.0, 1.0))


def simulate_fill(
    quoted_bid,
    quoted_ask,
    fair_value,
    quoted_spread,
    market_spread,
    volume,
    rng=None,
):
    """
    Simulate whether our bid or ask gets filled.

    Returns a dictionary with:
    - fill_side: 'buy', 'sell', or None
    - fill_price
    - spread_capture
    """

    if rng is None:
        rng = np.random.default_rng()

    fill_probability = estimate_fill_probability(
        quoted_spread=quoted_spread,
        market_spread=market_spread,
        volume=volume,
    )

    did_fill = rng.random() < fill_probability

    if not did_fill:
        return {
            "filled": False,
            "fill_side": None,
            "fill_price": np.nan,
            "fill_probability": fill_probability,
            "spread_capture": 0.0,
        }

    fill_side = rng.choice(["buy", "sell"])

    if fill_side == "buy":
        fill_price = quoted_bid
        spread_capture = fair_value - fill_price
    else:
        fill_price = quoted_ask
        spread_capture = fill_price - fair_value

    return {
        "filled": True,
        "fill_side": fill_side,
        "fill_price": fill_price,
        "fill_probability": fill_probability,
        "spread_capture": spread_capture,
    }