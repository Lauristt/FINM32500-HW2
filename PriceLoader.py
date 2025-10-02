##Implements Basic Functions to fetch 1)order price 2)order amount from a webscraper, and implements the PriceLoader class.
import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm
import time
import requests
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# CFG
START_DATE = "2005-01-01"
END_DATE = "2025-01-01"
DATA_DIR = "/Users/laurisli/Desktop/FINM32500/HW2/Data"
BATCH_SIZE = 50  # Number of tickers to fetch in one yfinance call
MIN_DATA_POINTS = 4000  # Expected max is around (2025-2005) * 252 trading days = ~5040
TICKER_LIST_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'



# task1: data acquisition
def fetch_sp500_tickers():
    print("Fetching S&P 500 ticker list from Wikipedia...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    # standard request with headers
    try:
        response = requests.get(TICKER_LIST_URL, headers=headers, timeout=10)
        response.raise_for_status()

        tables = pd.read_html(response.text)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()

        tickers = [t.replace('.', '-') for t in tickers]
        print(f"Found {len(tickers)} S&P 500 tickers.")
        return tickers

    # handle User-Agent or URL error
    except requests.exceptions.HTTPError as e:
        print(
            f" Error fetching tickers: HTTP Error {e.response.status_code}: {e.response.reason} - Please check User-Agent or URL。")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM']
    # other errors
    # ensure we at least return something if encountered patching error, i.e. code 403
    except Exception as e:
        print(f" Error fetching tickers: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM']

# download data, store them in .parquet format
def download_and_store_data(tickers, data_dir=DATA_DIR, start_date=START_DATE, end_date=END_DATE, batch_size=BATCH_SIZE,
                            min_data_points=MIN_DATA_POINTS):
    """
    Downloads historical adjusted close prices AND volume for tickers, batches requests,
    cleans data, and stores it in Parquet format.
    """
    print(f"\n Download started. (Range: {start_date} to {end_date}) ")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    successful_downloads = 0
    total_tickers = len(tickers)

    # Process tickers in batches
    for i in tqdm(range(0, total_tickers, batch_size), desc="Processing batches"):
        batch_tickers = tickers[i:i + batch_size]

        try:
            # yfinance download for a batch
            data = yf.download(
                tickers=batch_tickers,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,  # Automatically adjust close prices
                prepost=False,
                threads=True,
                proxy=None
            )

            if isinstance(data.columns, pd.MultiIndex):
                data_subset = data.loc[:, (['Close', 'Volume'], slice(None))]
            else:
                data_subset = data[['Close', 'Volume']]
            for ticker in batch_tickers:
                # robust design, pass tickers if not in batch
                if (ticker not in data_subset.columns.get_level_values(1).unique()) and (len(batch_tickers) > 1):
                    continue

                # Extract the DataFrame for the current ticker
                if len(batch_tickers) > 1 and isinstance(data_subset.columns, pd.MultiIndex):
                    ticker_data = data_subset.loc[:, (slice(None), ticker)]
                    # Rename columns from ('Close', Ticker) to 'Adj Close' and ('Volume', Ticker) to 'Volume'
                    ticker_data.columns = ['Adj Close', 'Volume']
                else:  # Single ticker case
                    ticker_data = data_subset.rename(columns={'Close': 'Adj Close'})

                # Clean up: drop rows where BOTH Adj Close and Volume are missing
                ticker_data = ticker_data.dropna(how='all')

                # Drop tickers with sparse or missing data (based on Adj Close count)
                if len(ticker_data['Adj Close'].dropna()) < min_data_points:
                    continue

                file_path = os.path.join(data_dir, f"{ticker}.parquet")
                ticker_data.to_parquet(file_path)
                successful_downloads += 1

        except Exception as e:
            print(f"Error when downloading and patching...Aborting... Error:{e}")
            pass

        # Respect API limits by sleeping briefly between batches
        time.sleep(1)

    print(f"\nData Download Complete.")
    print(
        f"Successfully downloaded and stored data for {successful_downloads}/{total_tickers} tickers. ({total_tickers - successful_downloads} tickers have been dropped due to incomplete series.)")


## task2: PriceLoader Class
class PriceLoader():
    """
    Manages the stock price database. loads both 'Adj Close' and 'Volume'.
    """
    def __init__(self, data_dir:str):
        self.data_dir = data_dir
        self._available_tickers = self._find_available_tickers()
        print(f"PriceLoader initialized. Found {len(self._available_tickers)} local price files.")

    def _find_available_tickers(self):
        """Identifies which tickers have a corresponding local data file."""
        if not os.path.exists(self.data_dir):
            return []

        files = os.listdir(self.data_dir)
        # Extract ticker from filename (e.g., 'AAPL.parquet' -> 'AAPL')
        tickers = [f.replace('.parquet', '') for f in files if f.endswith('.parquet')]
        return sorted(tickers)

    @property
    def available_tickers(self):
        """Returns a list of all tickers with locally stored data."""
        return self._available_tickers

    def load_data(self, ticker: str) -> pd.DataFrame | None:
        """
        Loads all stored data (Adj Close and Volume) for a specific ticker.
        Args:
            ticker: The stock ticker symbol.
        Returns:
            A pandas DataFrame with 'Adj Close' and 'Volume' columns, or None if the file is not found.
        """
        if ticker not in self._available_tickers:
            print(f" Error: Data for ticker '{ticker}' not found locally.")
            return None

        file_path = os.path.join(self.data_dir, f"{ticker}.parquet")
        try:
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            print(f"Error loading data for {ticker} from {file_path}: {e}")
            return None

    def load_price_data(self, ticker: str) -> pd.DataFrame | None:
        """
        Loads ONLY the adjusted close price data for a specific ticker.
        Keeps original method name for backward compatibility.
        """
        df = self.load_data(ticker)
        if df is not None:
            return df[['Adj Close']]
        return None

    def load_multiple_prices(self, tickers: list[str]) -> pd.DataFrame:
        """
        Loads and combines adjusted close price data for multiple tickers into a single DataFrame.
        Args:
            tickers: A list of stock ticker symbols.
        Returns:
            A wide-format pandas DataFrame with columns for each ticker's 'Adj Close' price.
        """
        price_data = {}
        for ticker in tickers:
            # Uses the modified load_data which returns a DataFrame with 'Adj Close' and 'Volume'
            df = self.load_data(ticker)
            if df is not None and 'Adj Close' in df.columns:
                price_data[ticker] = df['Adj Close']
        combined_df = pd.DataFrame(price_data)
        return combined_df


if __name__ == '__main__':

    sp500_tickers = fetch_sp500_tickers()
    print(f"Fetched sp500_tickers are:{sp500_tickers}")

    download_and_store_data(
        tickers=sp500_tickers,
        data_dir=DATA_DIR,
        start_date=START_DATE,
        end_date=END_DATE
    )
    loader = PriceLoader(DATA_DIR)

    # Test loading ALL data (new feature)
    sample_ticker = loader.available_tickers[0] if loader.available_tickers else 'AAPL'
    print(f"\nTest 1: Loading ALL data for single ticker: {sample_ticker}")
    data_all = loader.load_data(sample_ticker) # Use new load_data method
    if data_all is not None:
        print(f"Successfully loaded ALL {sample_ticker} data. Columns: {list(data_all.columns)}")
        print(data_all.head(3))

    # Test loading multiple tickers (Price only)
    sample_tickers = loader.available_tickers[:5]
    print(f"\nTest 2: Loading multiple tickers (Price only): {sample_tickers}")
    multi_data = loader.load_multiple_prices(sample_tickers)
    if not multi_data.empty:
        print(f"Successfully combined data. Shape: {multi_data.shape}")
        print(multi_data.head(3))