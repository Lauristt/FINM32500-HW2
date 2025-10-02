import pandas as pd
import numpy as np

try:
    from PriceLoader import PriceLoader
    from MovingAverageStrategy import Strategy
except ImportError as e:
    print(f"Fatal! Source Broken. Please implement source 'PriceLoader' or 'MovingAverageStrategy' (Baseclass). Error:{e}")

DATA_DIR = "/Users/laurisli/Desktop/FINM32500/HW2/Data"

class MACDStrategy(Strategy):
    """
    MACD Crossover Strategy.
    - Buy Signal: MACD line crosses above the signal line.
    - Sell Signal: MACD line crosses below the signal line.
    """
    def __init__(self, loader: PriceLoader, short_window=12, long_window=26, signal_window=9):
        super().__init__(loader)
        if short_window >= long_window:
            raise ValueError("Short EMA window must be smaller than long EMA window.")
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def generate_signals(self) -> pd.DataFrame:
        short_ema = self.prices.ewm(span=self.short_window, adjust=False).mean()
        long_ema = self.prices.ewm(span=self.long_window, adjust=False).mean()

        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=self.signal_window, adjust=False).mean()

        signals = (macd_line > signal_line).astype(int)
        return signals


if __name__ == "__main__":
    price_database = PriceLoader(data_dir=DATA_DIR)
    if not price_database.available_tickers:
        print("No data found. Exiting program.")
    else:
        macd_strategy = MACDStrategy(loader=price_database, short_window=12, long_window=26, signal_window=9)
        macd_strategy.run_backtest()