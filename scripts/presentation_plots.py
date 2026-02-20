import matplotlib.pyplot as plt
import numpy as np

from src.calibration import calibrate_t_fp, calibrate_t_levy_process
from src.data_loading import download_prices
from src.plotting import plot_histogram_and_t_fit, plot_profile_and_recent_price
from src.simulation import simulate_t_levy_process, tail_risk_estimate


def presentation_font_setup():
    """Set up slightly different matplotlib settings for presentation purposes."""
    plt.rcParams.update({
        "font.size": 12,  # larger than usual
        "figure.figsize": (6.4, 5.5)  # more square than usual 4:3 ratio
    })


def example_t_fit_to_NVDA():
    """
    Plot a histogram of NVDA daily log returns with a fitted t-distribution overlay.

    For this example, we are using equal weights for every data point for simplicity.
    """
    close_prices = download_prices(tickers=["NVDA"], period="3y", end_date="2025-05-01")["NVDA"]
    daily_log_returns = np.log(close_prices).diff().dropna()
    t_params = calibrate_t_fp(x=daily_log_returns, weights=np.ones_like(daily_log_returns))

    plt.figure()
    plot_histogram_and_t_fit(
        data=daily_log_returns,
        t_params=t_params,
        title=f"t-distribution fit to NVDA returns\n"
              f"from {min(close_prices.index.date)} to {max(close_prices.index.date)}\n"
              fr"df={t_params['df']:.2f}, $\mu={100 * t_params['mu']:.2f}\%$, $\sigma={100 * t_params['sigma']:.2f}\%$",
        plot_label="NVDA daily log returns",
        x_label="Log return"
    )


def example_t_levy_process_profiles():
    """
    Plot a historical price series along with future low and high profile projections.

    In this example, we show 6 months of recent prices and project out 3 months of future prices. Here, we are simply
    using uniform weights for the calibration since we have not introduced flexible probabilities yet.
    """
    close_prices = download_prices(tickers=["NVDA"], period="5y", end_date="2026-02-15")["NVDA"]
    daily_log_returns = np.log(close_prices).diff().dropna()
    calibration_date = "2025-05-01"

    params = calibrate_t_levy_process(daily_log_returns, end_date=calibration_date, num_years=3,
                                      tau_hl_df=None, tau_hl_sigma=None)
    np.random.seed(0)
    sim_paths = simulate_t_levy_process(
        # choosing a relatively high number of paths to get a smoother visualization of the profiles
        num_days=63, num_paths=100_000, S0=close_prices.loc[calibration_date],
        df=params['df'], mu=params['mu'], sigma=params['sigma']
    )
    simulated_profiles = tail_risk_estimate(simulation_paths=sim_paths, p=95, both_tails=True, return_profile=True)

    plt.figure()
    plot_profile_and_recent_price(close_prices, calibration_date, simulated_profiles,
                                  title="Example NVDA projection from\ncalibrated t-Levy process",
                                  recent_price_length=252 // 2, hide_prices_after_calibration=True)


if __name__ == "__main__":
    presentation_font_setup()

    example_t_fit_to_NVDA()
    example_t_levy_process_profiles()
    plt.show()
