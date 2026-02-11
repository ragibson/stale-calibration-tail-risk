import yfinance as yf


def download_prices_and_returns(tickers, period, end_date):
    """
    yfinance shim to download stock prices and returns for a set of tickers.

    :param tickers: List of tickers to download data for
    :param period: Length of time to download data for (e.g., "10y", "3mo", "1d")
    :param end_date: Final date to include ("YYYY-MM-DD" format)
    :return: DataFrame with stock prices and daily returns
    """
    close_prices = yf.download(tickers=tickers, period=period, end=end_date)["Close"].dropna()
    daily_returns = close_prices.pct_change()
    return close_prices, daily_returns
