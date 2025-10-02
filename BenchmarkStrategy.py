try:
    from PriceLoader import PriceLoader
except ImportError as e:
    print(f"Fatal! Source Broken. Please implement source 'PriceLoader'. Error:{e}")

import pandas as pd
import numpy as np

INITIAL_CASH = 1000000  # Initial cash: $1,000,000
# HW requirement: This keeps the individual trades small, avoiding market impact (slippage) as per Addendum 1.
MAX_DOLLAR_INVESTMENT_PER_TICKER = 5000 # Max dollar amount to invest per ticker in the initial purchase
DATA_DIR = "/Users/laurisli/Desktop/FINM32500/HW2/Data"

# Below used for sharpe calculation & transaction fees
TRANSACTION_COST_RATE = 0.001  # (one-sided, charged on purchase)
RISK_FREE_RATE = 0.04


class BenchmarkStrategy():
    def __init__(self, loader: PriceLoader):
        #self.prices:df, self.tickers:list
        self.loader = loader
        self.prices = self.loader.load_multiple_prices(self.loader.available_tickers)
        self.prices.dropna(inplace=True,how = 'all',axis = 0)
        self.tickers = self.prices.columns.tolist()

        #read attributes
        self.initial_cash = INITIAL_CASH
        self.cash =self.initial_cash
        # holdings: number of shares for each ticker
        self.holdings = {ticker:0 for ticker in self.tickers}
        # portfolio val: None-> pd.Series
        self.portfolio_value= None
        self.sharpe_ratio = None

    def run_backtest(self):
        """
        Executes the buy-and-hold benchmark strategy.

        Strategy (according to the HW Requirements):
        1. On the first available trading day, buy a fixed dollar amount of each stock,
           respecting the cash constraint.
        2. Hold all positions until the end of the period.
        3. Track the total portfolio value (cash + market value of stocks) daily.
        """
        print("Running benchmark strategy...")
        # purchase on day 1

        # robust design
        if self.prices.empty:
            print("Price data is empty. Cannot run backtest. Aborting...")
            raise ValueError("Price data is empty. Cannot run backtest.")

        first_day =self.prices.index[0]
        first_day_price = self.prices.loc[first_day].dropna()
        for ticker in self.tickers:
            if ticker in first_day_price and first_day_price[ticker]>0:
                # opt-in
                ticker_price = first_day_price[ticker]
                stock_position = MAX_DOLLAR_INVESTMENT_PER_TICKER//ticker_price
                # elif boundary cases, continue
                if stock_position==0:
                    continue
                cost=stock_position*(1+TRANSACTION_COST_RATE)*ticker_price

                # check if has enough cash
                if self.cash>=cost:
                    self.cash -=cost
                    self.holdings[ticker] +=stock_position
                else:
                    print(f"Skipping purchase of {ticker} on {first_day.date()}: insufficient cash.")
        holdings_series = pd.Series(self.holdings, dtype=float)
        market_value_over_time = self.prices.mul(holdings_series, axis = 'columns').sum(axis=1)
        self.portfolio_value = market_value_over_time+self.cash

        print("Benchmark strategy backtest complete.")
        print(f"Final cash remaining: ${self.cash:,.2f}")
        print(f"Initial portfolio value on {self.portfolio_value.index[0].date()}: ${self.portfolio_value.iloc[0]:,.2f}")
        print(f"Final portfolio value on {self.portfolio_value.index[-1].date()}: ${self.portfolio_value.iloc[-1]:,.2f}")

        self.calculate_performance_metrics()

    def calculate_performance_metrics(self):
        """
        Calculates key performance metrics like the Sharpe Ratio for the strategy.
        """
        if self.portfolio_value is None or len(self.portfolio_value) < 2:
            print("Portfolio value series not available or too short. Cannot calculate metrics.")
            return

        # Calculate daily returns from the portfolio value series
        daily_returns = self.portfolio_value.pct_change().dropna()
        excess_returns = daily_returns - (RISK_FREE_RATE / 252)

        # Calculate mean and standard deviation of excess returns
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()

        # Robust Design, Avoid division by zero
        if std_excess_return == 0:
            self.sharpe_ratio = np.inf if mean_excess_return > 0 else 0
        else:
            # 4. Calculate annualized Sharpe Ratio
            daily_sharpe_ratio = mean_excess_return / std_excess_return
            self.sharpe_ratio = daily_sharpe_ratio * np.sqrt(252)

        print(f"Annualized Sharpe Ratio: {self.sharpe_ratio:.4f}")


if __name__ == "__main__":
    Database = PriceLoader(DATA_DIR)
    benchmark_strategy = BenchmarkStrategy(Database)
    benchmark_strategy.run_backtest()





