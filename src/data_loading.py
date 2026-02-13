import pandas as pd
import yfinance as yf


def download_prices(tickers, period, end_date):
    """
    yfinance shim to download stock prices for a set of tickers.

    :param tickers: List of tickers to download data for
    :param period: Length of time to download data for (e.g., "10y", "3mo", "1d")
    :param end_date: Final date to include ("YYYY-MM-DD" format)
    :return: DataFrame with stock prices and daily returns
    """
    close_prices = yf.download(tickers=tickers, period=period, end=end_date)["Close"].dropna()
    return close_prices


def truncate_series_to_date_range(data, end_date, num_years):
    """
    Truncate a time series to a specified date range.

    :param data: DataFrame with a DateTime index to truncate
    :param end_date: Final date to include ("YYYY-MM-DD" format)
    :param num_years: Number of years to include in the truncated series
    :return: Truncated DataFrame
    """
    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.DateOffset(years=num_years)

    if data.index[0] > start_date:
        raise ValueError(f"Data starts at {data.index[0]}, which is after the desired start date of {start_date}.")

    return data.loc[start_date:end_date]
