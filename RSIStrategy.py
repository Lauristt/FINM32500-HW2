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
        print(
            f"Generating RSI signals (Window: {self.rsi_window}, Buy Threshold: >{self.oversold_threshold}, Sell Threshold: <{self.overbought_threshold})...")

    def generate_signals(self) -> pd.DataFrame:
        delta = self.prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Initialize signals DataFrame with NaNs
        signals = pd.DataFrame(np.nan, index=self.prices.index, columns=self.prices.columns)

        # Generate buy signals: RSI crosses above oversold threshold
        signals[rsi > self.oversold_threshold] = 1.0

        # Generate sell signals: RSI crosses below overbought threshold
        signals[rsi < self.overbought_threshold] = 0.0
        signals.ffill(inplace=True)
        signals.fillna(0, inplace=True)
        return signals.astype(int)


if __name__ == "__main__":
    price_database = PriceLoader(data_dir=DATA_DIR)
    if not price_database.available_tickers:
        print("No data found. Aborting...")
    else:
        rsi_strategy = RSIStrategy(loader=price_database, rsi_window=14, oversold_threshold=30, overbought_threshold=70)
        rsi_strategy.run_backtest()