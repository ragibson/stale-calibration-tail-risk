import matplotlib.pyplot as plt

from src.data_loading import download_prices_and_returns

if __name__ == "__main__":
    close_prices, daily_returns = download_prices_and_returns(
        tickers=["NVDA", "AAPL", "JPM", "KO"],
        period="10y",
        end_date="2026-01-30"
    )

    for ticker in close_prices.columns:
        plt.plot(close_prices.index, close_prices[ticker] / close_prices[ticker].iloc[0], label=ticker)
    plt.yscale("log")
    plt.legend()
    plt.show()
