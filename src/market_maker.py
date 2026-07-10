from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fill_sim import simulate_fill
from greeks import delta, vega
from inventory_manager import InventoryManager
from pnl import calculate_total_pnl
from quote_engine import generate_quotes


class MarketMaker:
    """
    Event-driven SPY options market-making simulator.

    The simulator:
    1. Samples sequential option quote updates.
    2. Generates adaptive bid/ask quotes.
    3. Simulates probabilistic fills.
    4. Updates inventory and cash.
    5. Recalculates portfolio risk and mark-to-market PnL.
    6. Stores a complete simulation history.
    """

    REQUIRED_COLUMNS = {
        "symbol",
        "underlying_price",
        "strike",
        "time_to_expiration",
        "risk_free_rate",
        "option_type",
        "implied_vol",
        "fair_value",
        "bid_ask_spread",
        "volume",
    }

    def __init__(
        self,
        option_data: pd.DataFrame,
        contract_multiplier: int = 100,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.option_data = option_data.copy().reset_index(drop=True)
        self.contract_multiplier = contract_multiplier
        self.rng = np.random.default_rng(random_seed)

        self.inventory = InventoryManager(
            contract_multiplier=contract_multiplier
        )

        self.history: List[Dict[str, Any]] = []
        self.current_prices: Dict[str, float] = {}

        self._validate_data()
        self._initialize_current_prices()

    def _validate_data(self) -> None:
        """
        Verify that the input DataFrame contains every field needed by
        the simulator.
        """

        missing_columns = (
            self.REQUIRED_COLUMNS - set(self.option_data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Option data is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if self.option_data.empty:
            raise ValueError("Option data cannot be empty.")

    def _initialize_current_prices(self) -> None:
        """
        Create one current fair-value mark for every option symbol.
        """

        self.current_prices = (
            self.option_data
            .drop_duplicates(subset="symbol", keep="last")
            .set_index("symbol")["fair_value"]
            .astype(float)
            .to_dict()
        )

    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """
        Calculate total delta and vega from all open option positions.

        Option Greek per contract
        × number of contracts
        × contract multiplier
        = portfolio exposure
        """

        net_delta = 0.0
        net_vega = 0.0

        positions = self.inventory.get_positions()

        for symbol, position in positions.items():
            if position == 0:
                continue

            contract_rows = self.option_data[
                self.option_data["symbol"] == symbol
            ]

            if contract_rows.empty:
                continue

            row = contract_rows.iloc[-1]

            option_delta = float(
                delta(
                    S=row["underlying_price"],
                    K=row["strike"],
                    T=row["time_to_expiration"],
                    r=row["risk_free_rate"],
                    sigma=row["implied_vol"],
                    option_type=row["option_type"],
                    q=0.0,
                )
            )

            option_vega = float(
                vega(
                    S=row["underlying_price"],
                    K=row["strike"],
                    T=row["time_to_expiration"],
                    r=row["risk_free_rate"],
                    sigma=row["implied_vol"],
                    q=0.0,
                )
            )

            net_delta += (
                position
                * self.contract_multiplier
                * option_delta
            )

            net_vega += (
                position
                * self.contract_multiplier
                * option_vega
            )

        return {
            "net_delta": net_delta,
            "net_vega": net_vega,
        }

    def process_update(
        self,
        update_number: int,
        row: pd.Series,
    ) -> Dict[str, Any]:
        """
        Process one quote-update event.
        """

        symbol = row["symbol"]

        portfolio_greeks_before = self.calculate_portfolio_greeks()

        current_position = self.inventory.get_position(symbol)

        quote = generate_quotes(
            fair_value=float(row["fair_value"]),
            market_spread=float(row["bid_ask_spread"]),
            implied_vol=float(row["implied_vol"]),
            inventory=current_position,
            net_delta=portfolio_greeks_before["net_delta"],
            net_vega=portfolio_greeks_before["net_vega"],
        )

        fill = simulate_fill(
            quoted_bid=quote["bid"],
            quoted_ask=quote["ask"],
            fair_value=float(row["fair_value"]),
            quoted_spread=quote["quoted_spread"],
            market_spread=float(row["bid_ask_spread"]),
            volume=float(row["volume"]),
            rng=self.rng,
        )

        if fill["filled"]:
            self.inventory.process_fill(
                symbol=symbol,
                fill_side=fill["fill_side"],
                fill_price=fill["fill_price"],
                quantity=1,
            )

        self.current_prices[symbol] = float(row["fair_value"])

        portfolio_greeks_after = self.calculate_portfolio_greeks()

        pnl_summary = calculate_total_pnl(
            cash=self.inventory.get_cash(),
            positions=self.inventory.get_positions(),
            current_prices=self.current_prices,
            contract_multiplier=self.contract_multiplier,
            initial_equity=0.0,
        )

        result = {
            "update_number": update_number,
            "symbol": symbol,
            "strike": float(row["strike"]),
            "option_type": row["option_type"],
            "fair_value": float(row["fair_value"]),
            "market_spread": float(row["bid_ask_spread"]),
            "quoted_bid": quote["bid"],
            "quoted_ask": quote["ask"],
            "quoted_spread": quote["quoted_spread"],
            "quote_skew": quote["quote_skew"],
            "fill_probability": fill["fill_probability"],
            "filled": fill["filled"],
            "fill_side": fill["fill_side"],
            "fill_price": fill["fill_price"],
            "spread_capture": fill["spread_capture"],
            "symbol_position": self.inventory.get_position(symbol),
            "net_delta": portfolio_greeks_after["net_delta"],
            "net_vega": portfolio_greeks_after["net_vega"],
            "cash": pnl_summary["cash"],
            "market_value": pnl_summary["market_value"],
            "total_equity": pnl_summary["total_equity"],
            "total_pnl": pnl_summary["total_pnl"],
        }

        self.history.append(result)

        return result

    def run(
        self,
        num_updates: int = 1000,
        sample_with_replacement: bool = True,
    ) -> pd.DataFrame:
        """
        Run a sequence of quote updates.

        When sample_with_replacement=True, contracts may appear repeatedly.
        This allows inventory to build, shrink, close, and reverse.
        """

        if num_updates <= 0:
            raise ValueError("num_updates must be positive.")

        if (
            not sample_with_replacement
            and num_updates > len(self.option_data)
        ):
            raise ValueError(
                "num_updates cannot exceed the number of rows when "
                "sampling without replacement."
            )

        sampled_indices = self.rng.choice(
            self.option_data.index,
            size=num_updates,
            replace=sample_with_replacement,
        )

        for update_number, row_index in enumerate(
            sampled_indices,
            start=1,
        ):
            row = self.option_data.loc[row_index]

            self.process_update(
                update_number=update_number,
                row=row,
            )

        return self.get_history()

    def get_history(self) -> pd.DataFrame:
        """
        Return the complete quote-update history.
        """

        return pd.DataFrame(self.history)

    def get_trade_log(self) -> pd.DataFrame:
        """
        Return all completed simulated fills.
        """

        return pd.DataFrame(self.inventory.get_trade_log())

    def get_positions(self) -> Dict[str, int]:
        """
        Return current option inventory.
        """

        return self.inventory.get_positions()

    def get_summary(self) -> Dict[str, Any]:
        """
        Return high-level simulation results.
        """

        history = self.get_history()

        if history.empty:
            return {
                "quote_updates": 0,
                "fills": 0,
                "fill_rate": 0.0,
                "total_spread_capture": 0.0,
                "final_pnl": 0.0,
                "final_net_delta": 0.0,
                "final_net_vega": 0.0,
            }

        fills = int(history["filled"].sum())
        quote_updates = len(history)

        return {
            "quote_updates": quote_updates,
            "fills": fills,
            "fill_rate": fills / quote_updates,
            "total_spread_capture": float(
                history["spread_capture"].sum()
                * self.contract_multiplier
            ),
            "final_pnl": float(history["total_pnl"].iloc[-1]),
            "final_cash": float(history["cash"].iloc[-1]),
            "final_market_value": float(
                history["market_value"].iloc[-1]
            ),
            "final_net_delta": float(
                history["net_delta"].iloc[-1]
            ),
            "final_net_vega": float(
                history["net_vega"].iloc[-1]
            ),
            "open_symbols": sum(
                position != 0
                for position in self.inventory.get_positions().values()
            ),
        }