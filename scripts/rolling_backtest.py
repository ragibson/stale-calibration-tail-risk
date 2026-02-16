import numpy as np
from tqdm import tqdm

from src.calibration import calibrate_t_levy_process, first_available_calibration_date
from src.data_loading import download_prices
from src.simulation import simulate_t_levy_process, tail_risk_estimate

if __name__ == "__main__":
    all_close_prices = download_prices(
        tickers=["NVDA", "AAPL", "LLY", "JPM", "KO"],
        period="10y",
        end_date="2026-01-30"
    )

    # we'll calibrate every 5 business days and projecting out the next period
    estimated_days = 5
    calibration_years = 3

    # we'll use a simple calculation of historical coverage rather than a statistical test
    all_num_samples = 0
    all_num_correct = 0
    for ticker in all_close_prices.columns:
        close_prices = all_close_prices[ticker]
        daily_log_returns = np.log(close_prices).diff().dropna()

        # determining the set of calibration dates
        first_calibration_date = first_available_calibration_date(daily_log_returns, num_years=calibration_years)
        calibration_dates = daily_log_returns.index[daily_log_returns.index >= first_calibration_date][::estimated_days]

        num_samples = 0
        num_correct = 0
        for calibration_date in tqdm(calibration_dates):
            params = calibrate_t_levy_process(daily_log_returns, end_date=calibration_date, num_years=calibration_years,
                                              tau_hl_df=252, tau_hl_sigma=252 // 4)

            # Monte Carlo simulation to estimate the left and right 95% confidence interval
            simulation_paths = simulate_t_levy_process(num_days=estimated_days, num_paths=10_000,
                                                       S0=close_prices.loc[calibration_date],
                                                       df=params['df'], mu=params['mu'], sigma=params['sigma'])
            process_confidence_interval = tail_risk_estimate(simulation_paths=simulation_paths, p=95,
                                                             both_tails=True, return_profile=False)

            # get the actual price 1 week after the calibration
            next_1week_index = close_prices.index.get_loc(calibration_date) + estimated_days
            if next_1week_index >= len(close_prices):
                continue
            next_1week_price = close_prices.iloc[next_1week_index]

            # record whether the actual price fell within our projected confidence interval
            num_samples += 1
            if process_confidence_interval[0] <= next_1week_price <= process_confidence_interval[1]:
                num_correct += 1

        print(f"Final '{ticker}' coverage: {num_correct / num_samples:.2%} over {num_samples} samples")
        all_num_samples += num_samples
        all_num_correct += num_correct

    print(f"Overall coverage: {all_num_correct / all_num_samples:.2%} over {all_num_samples} samples")
