import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

try:
    from PriceLoader import PriceLoader
except ImportError as e:
    print(f"Fatal! Source Broken. Please implement source 'PriceLoader'. Error:{e}")

DATA_DIR = "/Users/laurisli/Desktop/FINM32500/HW2/Data"
SHARE_PER_ORDER = 1
INITIAL_CASH = 1_000_000
RISK_FREE_RATE = 0.01  # Risk-free rate
TRANSACTION_COST_RATE = 0.001  # One-way transaction cost


class Strategy(ABC):
    """
    Vectorized Backtest Execution Engine, No Peeking Design (Baseclass).
    Ensure that all sub-strategies enforce the rule of using the previous
    day's signal for the current day's trade. This architecture prevents lookahead bias.
    """
    def __init__(self, loader: PriceLoader):
        self.loader = loader
        self.prices = self.loader.load_multiple_prices(self.loader.available_tickers)
        self.prices.dropna(inplace=True, how='all', axis=0)
        self.tickers = self.prices.columns.tolist()

        # Portfolio attributes
        self.initial_cash = INITIAL_CASH
        self.cash = self.initial_cash
        self.cash_over_time = []
        self.holdings = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = None
        self.sharpe_ratio = None
        self.holdings_df = pd.DataFrame(0, index=self.prices.index, columns=self.tickers)

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Returns:    A DataFrame with the same index and rows as
                    self.prices, where a value of 1 indicates a "long"
                    state and 0 indicates a "cash" state.
        """
        pass

    def run_backtest(self):
        start_time = time.time()
        print(f"Running backtest for {self.__class__.__name__}...")
        raw_signals = self.generate_signals()
        # CORE: the key line that ensures the validity of the backtest (No peeking).
        signals = raw_signals.shift(1)
        # Generate trade orders (1: buy, -1: sell).
        trades = signals.diff().fillna(0)
        # execute
        self._execute_trade_loop(trades)
        # report
        self._report_performance(start_time)

    def _execute_trade_loop(self, trades: pd.DataFrame):
        """
        Fully vectorized. Main trade execution logic
        """
        prices_np = self.prices.to_numpy()
        trades_np = trades.to_numpy()
        num_days, num_tickers = prices_np.shape

        self.holdings_df = pd.DataFrame(np.zeros_like(prices_np), index=self.prices.index, columns=self.prices.columns)
        holdings_np = self.holdings_df.to_numpy()
        self.cash_over_time = []

        current_holdings = np.zeros(num_tickers)

        # Main daily loop (necessary for stateful cash checks)
        for i in range(1, num_days):
            date = self.prices.index[i]
            # Get data for the current day
            daily_prices = prices_np[i]
            daily_trades = trades_np[i]
            sell_mask = (daily_trades == -1) & (current_holdings > 0)

            if np.any(sell_mask):
                shares_to_sell = current_holdings[sell_mask]
                prices_of_sells = daily_prices[sell_mask]
                proceeds = np.sum(shares_to_sell * prices_of_sells * (1 - TRANSACTION_COST_RATE))
                self.cash += proceeds
                current_holdings[sell_mask] = 0
            buy_mask = (daily_trades == 1) & (current_holdings == 0)
            buy_indices = np.where(buy_mask)[0] # np.where returns a tuple

            if len(buy_indices) > 0:
                for ticker_idx in buy_indices:
                    price_today = daily_prices[ticker_idx]
                    if not np.isnan(price_today):
                        cost = SHARE_PER_ORDER * price_today * (1 + TRANSACTION_COST_RATE)
                        if self.cash >= cost:
                            self.cash -= cost
                            current_holdings[ticker_idx] += SHARE_PER_ORDER

            holdings_np[i] = current_holdings
            self.cash_over_time.append(self.cash)

        self.cash_over_time = pd.Series(data=self.cash_over_time, index=self.prices.index[1:])
        self.holdings_df = pd.DataFrame(holdings_np, index=self.prices.index, columns=self.prices.columns)

        market_value_over_time = (self.holdings_df * self.prices).sum(axis=1)
        market_value_over_time = market_value_over_time.iloc[1:]  # Align with cash_series index

        self.portfolio_value = market_value_over_time + self.cash_over_time
        self.holdings = self.holdings_df.iloc[-1].to_dict()

    def _report_performance(self, start_time):
        """Print final backtest results. Private method"""
        print(f"{self.__class__.__name__} backtest complete.")
        print(f"Final cash remaining: ${self.cash:,.2f}")
        if self.portfolio_value is not None and not self.portfolio_value.empty:
            # Report the initial portfolio value correctly now
            initial_value = self.holdings_df.iloc[0].sum() * 0 + self.initial_cash  # Day 1 value is initial cash
            print(f"Initial portfolio value ({self.prices.index[0].date()}): ${initial_value:,.2f}")
            print(
                f"Final portfolio value ({self.portfolio_value.index[-1].date()}): ${self.portfolio_value.iloc[-1]:,.2f}")
            self.calculate_performance_metrics()
        else:
            print("Could not generate a valid portfolio value series.")

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    def calculate_performance_metrics(self):
        """Calculates and prints the annualized Sharpe Ratio.
        Report Style Design
        """
        if self.portfolio_value is None or len(self.portfolio_value) < 2:
            return
        daily_returns = self.portfolio_value.pct_change().dropna()
        excess_returns = daily_returns - (RISK_FREE_RATE / 252)
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()
        if std_excess_return == 0:
            sharpe_ratio = np.inf if mean_excess_return > 0 else 0
        else:
            sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)
        self.sharpe_ratio = sharpe_ratio
        print(f"Annualized Sharpe Ratio: {self.sharpe_ratio:.4f}")

    def _get_sharpe_drawdown(self):
        """
        Calculates the Sharpe Ratio and maximal draw down.
        :return: sharpe ratio: float and maximal draw down: float
        """
        if len(self.cash_over_time) < 2:
            return None, None
        portfolio_value = pd.Series(self.cash_over_time)
        # daily return
        daily_returns = portfolio_value.pct_change().dropna()
        excess_returns = daily_returns - (RISK_FREE_RATE / 252)
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()
        if std_excess_return == 0:
            sharpe_ratio = np.inf if mean_excess_return > 0 else 0
        else:
            sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)
        # max drawdown
        cumulative_max = portfolio_value.cummax()
        drawdowns = (portfolio_value - cumulative_max) / cumulative_max
        max_drawdown = drawdowns.min()
        return sharpe_ratio, max_drawdown

    @property
    def _holdings_sanity_check(self):
        """Sanity check to ensure no short positions were created."""
        return (self.holdings_df >= 0).all().all()


# HW: Moving Average Strategy Implementation
class MovingAverageStrategy(Strategy):
    """
    Moving Average Crossover Strategy.
    - Buy Signal: Short-term MA > Long-term MA
    - Sell Signal: Short-term MA <= Long-term MA
    """
    def __init__(self, loader: PriceLoader, short_window=20, long_window=50):
        super().__init__(loader)
        if short_window >= long_window:
            raise ValueError("Short window must be smaller than long window.")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        short_ma = self.prices.rolling(window=self.short_window).mean()
        long_ma = self.prices.rolling(window=self.long_window).mean()
        signals = (short_ma > long_ma).astype(int)
        return signals


if __name__ == "__main__":
    price_database = PriceLoader(data_dir=DATA_DIR)
    if not price_database.available_tickers:
        print("No data found. Aborting...")
    else:
        ma_strategy = MovingAverageStrategy(loader=price_database, short_window=20, long_window=50)
        ma_strategy.run_backtest()