from typing import Dict, Mapping, Any


def calculate_market_value(
    positions: Mapping[str, int],
    current_prices: Mapping[str, float],
    contract_multiplier: int = 100,
) -> float:
    """
    Calculate the marked market value of all open option positions.

    Long positions have positive market value.
    Short positions have negative market value.
    """

    market_value = 0.0

    for symbol, position in positions.items():
        if symbol not in current_prices:
            raise KeyError(f"No current price found for {symbol}")

        market_value += (
            position
            * float(current_prices[symbol])
            * contract_multiplier
        )

    return market_value


def calculate_total_pnl(
    cash: float,
    positions: Mapping[str, int],
    current_prices: Mapping[str, float],
    contract_multiplier: int = 100,
    initial_equity: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate mark-to-market portfolio value and total PnL.

    total_equity = cash + market value of open positions
    total_pnl = total_equity - initial equity
    """

    market_value = calculate_market_value(
        positions=positions,
        current_prices=current_prices,
        contract_multiplier=contract_multiplier,
    )

    total_equity = float(cash) + market_value
    total_pnl = total_equity - float(initial_equity)

    return {
        "cash": float(cash),
        "market_value": market_value,
        "total_equity": total_equity,
        "total_pnl": total_pnl,
    }

def calculate_unrealized_pnl(
    positions,
    average_prices,
    current_prices,
    contract_multiplier=100,
):
    """
    Calculate unrealized PnL for all open positions.

    Long:
        (current price - average price) × position × multiplier

    Short:
        The signed position automatically reverses the direction.
    """

    unrealized_pnl = 0.0

    for symbol, position in positions.items():
        if position == 0:
            continue

        if symbol not in current_prices:
            raise KeyError(f"No current price found for {symbol}")

        if symbol not in average_prices:
            raise KeyError(f"No average price found for {symbol}")

        unrealized_pnl += (
            position
            * (
                float(current_prices[symbol])
                - float(average_prices[symbol])
            )
            * contract_multiplier
        )

    return unrealized_pnl