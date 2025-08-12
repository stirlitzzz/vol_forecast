import pandas as pd
from datetime import date, datetime
from .time_utils import session_close_utc, to_utc
from .ledger import TradeLedger

# =========================
# Portfolio snapshot
# =========================
class PortfolioView:
    def __init__(self, ledger: TradeLedger):
        self.ledger = ledger

    def positions_at(self, as_of) -> pd.DataFrame:
        """
        as_of: date or datetime. If date, uses NY close of that date.
        Returns aggregated open positions (UTC compare).
        """
        if isinstance(as_of, date) and not isinstance(as_of, datetime):
            as_of_utc = session_close_utc(as_of)
        else:
            as_of_utc = to_utc(as_of)

        df = self.ledger.trades_df
        if df.empty:
            return df

        df = df[df["timestamp"] <= as_of_utc]
        grouped = (
            df.groupby(["instrument_type","underlying","option_type","strike","expiry"],
                       dropna=False)["signed_quantity"].sum().reset_index()
        )
        return grouped[grouped["signed_quantity"] != 0]
