from typing import Dict, Any


class InventoryManager:
    """
    Tracks option positions, cash, weighted-average entry prices,
    realized PnL, and a complete trade log.
    """

    def __init__(self, contract_multiplier: int = 100):
        self.contract_multiplier = contract_multiplier

        self.positions: Dict[str, int] = {}
        self.average_prices: Dict[str, float] = {}
        self.realized_pnl: Dict[str, float] = {}

        self.cash = 0.0
        self.trade_log = []

    def process_fill(
        self,
        symbol: str,
        fill_side: str,
        fill_price: float,
        quantity: int = 1,
    ) -> None:
        """
        Process a buy or sell fill and update inventory accounting.

        Parameters
        ----------
        symbol:
            Unique option identifier.

        fill_side:
            'buy' or 'sell'.

        fill_price:
            Option premium per share.

        quantity:
            Number of option contracts.
        """

        if fill_side not in {"buy", "sell"}:
            raise ValueError("fill_side must be 'buy' or 'sell'.")

        if fill_price <= 0:
            raise ValueError("fill_price must be positive.")

        if quantity <= 0:
            raise ValueError("quantity must be positive.")

        old_position = self.positions.get(symbol, 0)
        old_average_price = self.average_prices.get(symbol, 0.0)

        signed_quantity = quantity if fill_side == "buy" else -quantity
        new_position = old_position + signed_quantity

        cash_change = (
            -signed_quantity
            * fill_price
            * self.contract_multiplier
        )

        realized_change = 0.0

        # Case 1: Opening a completely new position
        if old_position == 0:
            new_average_price = fill_price

        # Case 2: Increasing an existing position in the same direction
        elif old_position * signed_quantity > 0:
            total_old_cost = abs(old_position) * old_average_price
            new_trade_cost = abs(signed_quantity) * fill_price

            new_average_price = (
                total_old_cost + new_trade_cost
            ) / abs(new_position)

        # Case 3: Reducing, closing, or reversing a position
        else:
            closing_quantity = min(
                abs(old_position),
                abs(signed_quantity),
            )

            if old_position > 0:
                # Closing a long position by selling
                realized_change = (
                    fill_price - old_average_price
                ) * closing_quantity * self.contract_multiplier

            else:
                # Closing a short position by buying
                realized_change = (
                    old_average_price - fill_price
                ) * closing_quantity * self.contract_multiplier

            if new_position == 0:
                # Position fully closed
                new_average_price = 0.0

            elif old_position * new_position > 0:
                # Partial close; remaining inventory keeps old cost basis
                new_average_price = old_average_price

            else:
                # Trade crossed through zero and opened opposite position
                new_average_price = fill_price

        self.positions[symbol] = new_position
        self.average_prices[symbol] = new_average_price
        self.cash += cash_change

        previous_realized = self.realized_pnl.get(symbol, 0.0)
        self.realized_pnl[symbol] = previous_realized + realized_change

        self.trade_log.append({
            "symbol": symbol,
            "side": fill_side,
            "quantity": quantity,
            "fill_price": fill_price,
            "cash_change": cash_change,
            "old_position": old_position,
            "new_position": new_position,
            "average_price": new_average_price,
            "realized_pnl_change": realized_change,
            "cumulative_realized_pnl": self.realized_pnl[symbol],
            "cash": self.cash,
        })

    def get_position(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)

    def get_positions(self) -> Dict[str, int]:
        return self.positions.copy()

    def get_average_price(self, symbol: str) -> float:
        return self.average_prices.get(symbol, 0.0)

    def get_average_prices(self) -> Dict[str, float]:
        return self.average_prices.copy()

    def get_realized_pnl(self, symbol: str | None = None) -> float:
        if symbol is not None:
            return self.realized_pnl.get(symbol, 0.0)

        return float(sum(self.realized_pnl.values()))

    def get_cash(self) -> float:
        return self.cash

    def get_trade_log(self):
        return self.trade_log.copy()

    def get_position_details(self) -> Dict[str, Dict[str, Any]]:
        symbols = set(self.positions) | set(self.average_prices)

        return {
            symbol: {
                "position": self.positions.get(symbol, 0),
                "average_price": self.average_prices.get(symbol, 0.0),
                "realized_pnl": self.realized_pnl.get(symbol, 0.0),
            }
            for symbol in symbols
        }