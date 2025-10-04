import pandas as pd
import numpy as np

try:
    from PriceLoader import PriceLoader
    from MovingAverageStrategy import Strategy
except ImportError as e:
    print(f"Fatal! Source Broken. Please implement source 'PriceLoader' or 'MovingAverageStrategy' (Baseclass). Error:{e}")


DATA_DIR = "/Users/laurisli/Desktop/FINM32500/HW2/Data"


class RSIStrategy(Strategy):
    """
    RSI Crossover Strategy (Long Only).
    - Buy Signal: RSI crosses above the oversold threshold.
    - Sell Signal: RSI crosses below the overbought threshold.
    """

    def __init__(self, loader: PriceLoader, rsi_window=14, oversold_threshold=30, overbought_threshold=70):
        super().__init__(loader)
        self.rsi_window = rsi_window
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def generate_signals(self) -> pd.DataFrame:
        delta = self.prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/self.rsi_window, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/self.rsi_window, adjust=False).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signals = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        signals[rsi < self.oversold_threshold] = 1.0

        return signals.astype(int)

if __name__ == "__main__":
    price_database = PriceLoader(data_dir=DATA_DIR)
    if not price_database.available_tickers:
        print("No data found. Aborting...")
    else:
        rsi_strategy = RSIStrategy(loader=price_database, rsi_window=14, oversold_threshold=30, overbought_threshold=70)
        rsi_strategy.run_backtest()