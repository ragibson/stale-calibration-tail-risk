import matplotlib.pyplot as plt
import numpy as np

from src.calibration import calibrate_t_fp
from src.data_loading import download_prices
from src.plotting import plot_histogram_and_t_fit


def example_t_fit_to_NVDA():
    """
    Plot a histogram of NVDA daily log returns with a fitted t-distribution overlay.

    For this example, we are using equal weights for every data point for simplicity.
    """
    close_prices = download_prices(
        tickers=["NVDA"],
        period="3y",
        end_date="2026-01-30"
    )["NVDA"]
    daily_log_returns = np.log(close_prices).diff().dropna()

    t_params = calibrate_t_fp(
        x=daily_log_returns,
        weights=np.ones_like(daily_log_returns)
    )

    plot_histogram_and_t_fit(
        data=daily_log_returns,
        t_params=t_params,
        title=f"t-distribution fit to NVDA returns from "
              f"{min(close_prices.index.date)} to {max(close_prices.index.date)}\n"
              fr"df={t_params['df']:.2f}, $\mu={t_params['mu']:.4f}$, $\sigma={t_params['sigma']:.4f}$",
        plot_label="NVDA daily log returns",
        x_label="Log return"
    )
    plt.show()


if __name__ == "__main__":
    example_t_fit_to_NVDA()
