class InventoryManager:
    """
    Tracks option inventory and cash after simulated fills.
    """

    def __init__(self, contract_multiplier=100):
        self.contract_multiplier = contract_multiplier
        self.positions = {}
        self.cash = 0.0
        self.trade_log = []

    def process_fill(self, symbol, fill_side, fill_price, quantity=1):
        """
        Updates position and cash after a fill.

        fill_side:
        - 'buy' means we bought the option
        - 'sell' means we sold the option
        """

        if fill_side not in ["buy", "sell"]:
            return

        if symbol not in self.positions:
            self.positions[symbol] = 0

        signed_quantity = quantity if fill_side == "buy" else -quantity
        cash_change = -signed_quantity * fill_price * self.contract_multiplier

        self.positions[symbol] += signed_quantity
        self.cash += cash_change

        self.trade_log.append({
            "symbol": symbol,
            "side": fill_side,
            "quantity": quantity,
            "fill_price": fill_price,
            "cash_change": cash_change,
            "new_position": self.positions[symbol],
            "cash": self.cash,
        })

    def get_position(self, symbol):
        return self.positions.get(symbol, 0)

    def get_positions(self):
        return self.positions

    def get_cash(self):
        return self.cash

    def get_trade_log(self):
        return self.trade_log