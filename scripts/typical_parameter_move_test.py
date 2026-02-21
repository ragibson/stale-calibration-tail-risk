import numpy as np
from tqdm import tqdm

from src.calibration import calibrate_t_levy_process, first_available_calibration_date
from src.data_loading import download_prices

if __name__ == "__main__":
    close_prices = download_prices(
        tickers=["AAPL"],
        period="10y",
        end_date="2026-01-30"
    )
    daily_log_returns = np.log(close_prices).diff().dropna()

    # rolling monthly calibrations over the time series
    calibration_years = 3
    first_calibration_date = first_available_calibration_date(daily_log_returns, num_years=calibration_years)
    # recalibrating every 21 business days (~1 month)
    calibration_dates = daily_log_returns.index[daily_log_returns.index >= first_calibration_date][::21]

    calibrated_params = [
        calibrate_t_levy_process(daily_log_returns, end_date=calibration_date, num_years=calibration_years)
        for calibration_date in tqdm(calibration_dates)
    ]

    # average df decrease and mu/sigma increase (conditional on those moves actually happening)
    average_df_decrease = np.mean([calibrated_params[i]['df'] - calibrated_params[i - 1]['df']
                                     for i in range(1, len(calibrated_params))
                                     if calibrated_params[i]['df'] < calibrated_params[i - 1]['df']])
    average_mu_increase = np.mean([calibrated_params[i]['mu'] - calibrated_params[i - 1]['mu']
                                     for i in range(1, len(calibrated_params))
                                     if calibrated_params[i]['mu'] > calibrated_params[i - 1]['mu']])
    average_sigma_increase = np.mean([calibrated_params[i]['sigma'] - calibrated_params[i - 1]['sigma']
                                        for i in range(1, len(calibrated_params))
                                        if calibrated_params[i]['sigma'] > calibrated_params[i - 1]['sigma']])

    print("Typical parameter moves on monthly AAPL recalibrations:")
    print(f"Average df decrease: {average_df_decrease:.4f} (conditional on df decrease)")
    print(f"Average mu increase: {average_mu_increase:.4%} (conditional on mu increase)")
    print(f"Average sigma increase: {average_sigma_increase:.4%} (conditional on sigma increase)")
