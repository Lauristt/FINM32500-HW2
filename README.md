#  FINM32500-HW2: Quantitative Trading Strategy Comparison

A comprehensive framework for implementing, backtesting, and comparing various algorithmic trading strategies using Python and Jupyter Notebooks.

Author: Yuting Li, Xiangchen Liu, Rajdeep Choudhury, Simon Guo


##  Features

*    **Multiple Strategy Implementations**: Explore and utilize pre-built strategies including Moving Average Crossover, MACD, RSI, and Volatility Breakout.
*    **Modular Design**: Each strategy is encapsulated in its own Python file, promoting reusability and easy extension.
*    **Interactive Comparison**: Utilize Jupyter Notebooks (`StrategyComparison.ipynb`) for interactive analysis, backtesting, and visual comparison of strategy performance.
*    **Robust Price Data Loader**: `PriceLoader.py` provides a reliable mechanism to fetch and manage historical financial instrument data.
*    **Benchmark Performance**: Includes a `BenchmarkStrategy.py` to compare custom strategy returns against a simple market benchmark.


##  Installation Guide

To get started with this project, follow these steps to set up your environment and dependencies.

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/Lauristt/FINM32500-HW2.git
cd FINM32500-HW2
```

### 2. Create a Virtual Environment (Recommended)

It is highly recommended to create a virtual environment to manage project dependencies:

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install pandas numpy matplotlib yfinance jupyterlab
```

### 4. Run Jupyter Notebook

Launch Jupyter Lab to open and interact with the strategy comparison notebook:

```bash
jupyter lab
```

Your web browser should open to the Jupyter Lab interface. Navigate to `StrategyComparison.ipynb` and open it.


## Usage Examples

The core of this project is the `StrategyComparison.ipynb` notebook, which allows you to load data, apply different strategies, and visualize their performance.

### Running Strategy Comparison

1.  **Open `StrategyComparison.ipynb`** in Jupyter Lab.
2.  **Fetch Data**: The notebook uses `PriceLoader.py` to fetch historical stock data. You can modify the ticker symbol and date range within the notebook.
    ```python
    import pandas as pd
    from PriceLoader import PriceLoader

    # Example: Fetching data for Apple (AAPL)
    loader = PriceLoader()
    df = loader.get_data(ticker="AAPL", start_date="2020-01-01", end_date="2023-01-01")
    print(df.head())
    ```
3.  **Apply Strategies**: Instantiate and apply the desired strategies to your loaded data.
    ```python
    from MACDStrategy import MACDStrategy
    from MovingAverageStrategy import MovingAverageStrategy
    # ... import other strategies

    # Example: Applying MACD Strategy
    macd_strategy = MACDStrategy(df)
    macd_strategy.generate_signals()
    macd_returns = macd_strategy.calculate_returns()

    # Example: Applying Moving Average Strategy
    ma_strategy = MovingAverageStrategy(df, short_window=20, long_window=50)
    ma_strategy.generate_signals()
    ma_returns = ma_strategy.calculate_returns()
    ```
4.  **Analyze and Compare**: The notebook will guide you through visualizing cumulative returns, drawdowns, and other performance metrics for each strategy.

Feel free to modify strategy parameters, add new strategies, or experiment with different financial instruments directly within the notebook.


## Project Roadmap

This project is continuously evolving. Here are some planned enhancements and future goals:

*   **Advanced Backtesting Framework**: Integrate a more sophisticated backtesting engine with transaction costs, slippage, and position sizing.
*   **More Trading Strategies**: Implement additional popular quantitative strategies (e.g., ARIMA, Machine Learning-based strategies).
*   **Performance Metrics Dashboard**: Develop an interactive dashboard for a richer comparison of strategy performance metrics.
*   **Optimization Modules**: Add modules for hyperparameter optimization of strategy parameters.
*   **Cloud Integration**: Explore options for fetching data from cloud sources or deploying strategies to cloud platforms.


## Contribution Guidelines

We welcome contributions to enhance this project! Please follow these guidelines:

*   **Fork the Repository**: Start by forking the `FINM32500-HW2` repository to your GitHub account.
*   **Create a Feature Branch**: Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
*   **Code Style**: Adhere to PEP 8 for Python code. Use clear, concise variable names and add comments where necessary.
*   **Commit Messages**: Write clear and descriptive commit messages.
*   **Testing**: If you're adding new strategies or significant features, please include relevant unit tests in the `test_data_fetch.py` or a new test file.
*   **Pull Request Process**:
    1.  Ensure your code is well-tested and passes all existing checks.
    2.  Open a Pull Request (PR) to the `main` branch of the original repository.
    3.  Provide a detailed description of your changes in the PR.


##  License

This project is protected under the MIT LICENSE. For more details, refer to the LICENSE file.

##  Acknowledgments

This project was created as part of the FINM 32500 course at The University of Chicago. Inspiration from various open-source backtesting frameworks.
