import pandas as pd
import numpy as np

try:
    from PriceLoader import PriceLoader
    from MovingAverageStrategy import Strategy
except ImportError as e:
    print(f"Fatal! Source Broken. Please implement source 'PriceLoader' or 'MovingAverageStrategy' (Baseclass). Error:{e}")

DATA_DIR = "/Users/laurisli/Desktop/FINM32500/HW2/Data"
# HW: Vol breakout Strategy Implementation
class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout Strategy.
    - Position is entered (Buy) when rolling_std > daily_returns.
    - Position is exited (Sell) when rolling_std <= daily_returns.
    """
    def __init__(self, loader: PriceLoader, cal_std_window=20):
        super().__init__(loader)
        self.cal_std_window = cal_std_window

    def generate_signals(self) -> pd.DataFrame:
        daily_returns = self.prices.pct_change()
        rolling_std = self.prices.rolling(window=self.cal_std_window).std()
        signals = (rolling_std > daily_returns).astype(int)
        return signals


if __name__ == "__main__":
    price_database = PriceLoader(data_dir=DATA_DIR)
    if not price_database.available_tickers:
        print("No data found. Aborting...")
    else:
        vb_strategy = VolatilityBreakoutStrategy(loader=price_database, cal_std_window=20)
        vb_strategy.run_backtest()