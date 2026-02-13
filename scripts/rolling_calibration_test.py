import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.calibration import calibrate_t_levy_process
from src.data_loading import download_prices

if __name__ == "__main__":
    close_prices = download_prices(
        tickers=["AAPL"],
        period="10y",
        end_date="2026-01-30"
    )
    daily_log_returns = np.log(close_prices).diff().dropna()

    # rolling calibration over the time series
    calibration_dates = close_prices.index[252 * 3 + 1::5]
    calibrated_params = [calibrate_t_levy_process(daily_log_returns, end_date=calibration_date, num_years=3)
                         for calibration_date in tqdm(calibration_dates)]

    # plotting the calibrated parameters over time
    fig, ax = plt.subplots(nrows=3, figsize=(6, 12), sharex=True)

    for i, param_name in enumerate(calibrated_params[0].keys()):
        param_values = [params[param_name] for params in calibrated_params]
        ax[i].plot(calibration_dates, param_values, label=param_name)

        ax_price = ax[i].twinx()
        ax_price.plot(close_prices.index, close_prices, alpha=0.5, color='black')

        ax[i].set_ylabel(param_name)
        ax_price.set_ylabel("Price")
        ax_price.set_yscale("log")

        ax[i].legend()
        ax[i].set_title(f"Rolling Calibration of {param_name} with Price Overlay")

    plt.tight_layout()
    plt.show()
