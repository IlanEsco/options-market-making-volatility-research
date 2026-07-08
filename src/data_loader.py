import pandas as pd


REQUIRED_COLUMNS = [
    "quote_date",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "underlying_price",
    "risk_free_rate",
]


def load_options_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["bid_ask_spread"] = df["ask"] - df["bid"]
    df["time_to_expiration"] = (
        df["expiration"] - df["quote_date"]
    ).dt.days / 365

    df["moneyness"] = df["strike"] / df["underlying_price"]

    df = df[df["bid"] >= 0]
    df = df[df["ask"] > df["bid"]]
    df = df[df["mid_price"] > 0]
    df = df[df["time_to_expiration"] > 0]

    return df