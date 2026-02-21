import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from tqdm import tqdm

from src.backtesting import stale_calibration_comparison
from src.calibration import (calibrate_t_fp, calibrate_t_levy_process, exponential_decay_probabilities,
                             first_available_calibration_date)
from src.data_loading import download_prices
from src.plotting import plot_histogram_and_t_fit, plot_profile_and_recent_price, plot_parameter_stability_and_log_price
from src.simulation import simulate_t_levy_process, tail_risk_estimate


def presentation_font_setup():
    """Set up slightly different matplotlib settings for presentation purposes."""
    plt.rcParams.update({
        "font.size": 12,  # larger than usual
        "figure.figsize": (6.4, 5.5),  # more square than usual 4:3 ratio
        "figure.dpi": 150
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
        title=f"$t$-distribution fit to NVDA returns\n"
              f"from {min(close_prices.index.date)} to {max(close_prices.index.date)}\n"
              fr"$\mu={100 * t_params['mu']:.2f}\%$, $\sigma={100 * t_params['sigma']:.2f}\%$, "
              fr"$\nu={t_params['df']:.2f}$",
        plot_label="NVDA daily log returns",
        x_label="Log return"
    )


def example_t_levy_process_profiles():
    """
    Plot a historical price series along with future low and high profile projections.

    In this example, we show 6 months of recent prices and project out 3 months of future prices.

    Note that we are using exponential decay weights even though it has not yet been discussed in the presentation.
    """
    close_prices = download_prices(tickers=["NVDA"], period="5y", end_date="2026-02-15")["NVDA"]
    daily_log_returns = np.log(close_prices).diff().dropna()
    calibration_date = "2025-05-01"

    params = calibrate_t_levy_process(daily_log_returns, end_date=calibration_date, num_years=3,
                                      tau_hl_df=252, tau_hl_sigma=252 // 4)
    np.random.seed(0)
    sim_paths = simulate_t_levy_process(
        # choosing a relatively high number of paths to get a smoother visualization of the profiles
        num_days=63, num_paths=100_000, S0=close_prices.loc[calibration_date],
        df=params['df'], mu=params['mu'], sigma=params['sigma']
    )
    simulated_profiles = tail_risk_estimate(simulation_paths=sim_paths, p=95, both_tails=True, return_profile=True)

    plt.figure()
    plot_profile_and_recent_price(close_prices, calibration_date, simulated_profiles,
                                  title="Example NVDA projection from\ncalibrated $t$-Levy process",
                                  recent_price_length=252 // 2, hide_prices_after_calibration=True)


def example_exponential_decay_probabilities():
    """
    Plot impact of exponential decay probabilities over time.

    Note this plot is so simple that it is not included in the plotting module.
    """
    tau_hl_df = 252
    tau_hl_sigma = 252 // 4
    calibration_days = 252 * 3  # ~3 years

    df_weights = exponential_decay_probabilities(tau_hl=tau_hl_df, t_bar=calibration_days)
    sigma_weights = exponential_decay_probabilities(tau_hl=tau_hl_sigma, t_bar=calibration_days)
    calibration_period_yearfracs = np.linspace(-3, 0, calibration_days)  # years before calibration date

    plt.figure()
    plt.plot(calibration_period_yearfracs, df_weights,
             label=r"Weights for $\nu$ calibration, $\tau_{HL}=$252 days")
    plt.plot(calibration_period_yearfracs, sigma_weights,
             label=r"Weights for $\mu$ and $\sigma$ calibration, $\tau_{HL}=$63 days")
    plt.title("Exponential decay probabilities for\n3-year $t$-Levy process calibration")
    plt.ylabel("Weight")
    plt.xlabel("Years before calibration date")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend()
    plt.grid()
    plt.tight_layout()


def example_parameter_stability():
    """Plot the stability of calibrated t-Levy process parameters over time (with log-price overlay)."""
    ticker = "LLY"
    close_prices = download_prices(
        tickers=[ticker],
        period="10y",
        end_date="2026-01-30"
    )
    daily_log_returns = np.log(close_prices).diff().dropna()

    # rolling calibration over the time series
    calibration_years = 3
    first_calibration_date = first_available_calibration_date(daily_log_returns, num_years=calibration_years)
    # recalibrating every ~5 business days to speed up this plot
    calibration_dates = daily_log_returns.index[daily_log_returns.index >= first_calibration_date][::5]

    calibrated_params = [
        calibrate_t_levy_process(daily_log_returns, end_date=calibration_date, num_years=calibration_years)
        for calibration_date in tqdm(calibration_dates, desc="Rolling calibration of t-Levy process")
    ]

    params_to_plot = ('df', 'sigma')
    param_latex = (r'$\nu$', r'$\sigma$')
    plot_parameter_stability_and_log_price(
        close_prices=close_prices, calibration_dates=calibration_dates, calibrated_params=calibrated_params,
        params_to_plot=params_to_plot, param_latex=param_latex, figsize=(7, 13),
        plot_titles=tuple(rf"Rolling {ticker} 3-year calibration of {p} with price overlay"
                          for p in param_latex)
    )


def example_stale_calibration_backtest_p_values():
    """
    Plot an example of backtesting a stale calibration with a newly recalibrated t-Levy process.

    The corresponding p-values are pushed towards 0, representing an increased likelihood of failing a statistical test.
    """
    # df and sigma taken from AAPL's COVID crash, changes from 3/9/2020 to 4/14/2020
    stale_params = {'df': 2.9533, 'mu': 0.0, 'sigma': 0.010604}
    new_params = {'df': 2.4844, 'mu': 0.0, 'sigma': 0.010894}

    # mimicking backtesting of 2-week returns with a full year of data (non-overlapping windows)
    p_values = stale_calibration_comparison(
        stale_t_params=stale_params, new_t_params=new_params, num_days=10, num_samples_per_trial=25,
        num_trials=10_000, num_paths=10_000, confidence_level=0.95
    )

    # using a bar chart here instead of a histogram since these ranges are close to categories than continuous values
    histogram_bins = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 1.0]
    counts, _ = np.histogram(p_values, bins=histogram_bins)
    counts = counts / counts.sum()
    x_locations = np.arange(len(counts))
    plt.bar(x_locations, counts, edgecolor="black")
    plt.xticks(x_locations, ["<1%", "1%-5%", "5–25%", "25–50%", "50-75%", "75–100%"])

    plt.title("AAPL backtest between two calibrations\n"
              rf"Stale (March 2020): $\nu={stale_params['df']:.2f}$, $\sigma={100 * stale_params['sigma']:.2f}\%$"
              "\n"
              rf"New (April 2020): $\nu={new_params['df']:.2f}$, $\sigma={100 * new_params['sigma']:.2f}\%$")
    plt.ylabel("Percent of trials")
    plt.xlabel("Backtest $p$-value")

    plt.yticks(np.arange(0.0, plt.ylim()[1], 0.025))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.tight_layout()


def example_stale_calibration_degradation():
    """
    Plot a very rough estimate of backtest degradation as parameters drift over time.

    These particular parameter trends are taken from AAPL's historical data, and the starting parameters are arbitrarily
    chosen to be representative of something like a 2022-era AAPL.

    This is a ballpark estimate assuming a lot of independence! We're assuming the parameters trend against us,
    calibrated parameters move independently and those moves can be extrapolated via average historical trends, etc.

    Moreover, everything is relative to the starting point of the stale calibration!
    """
    stale_params = {'df': 4.0, 'mu': 0.001, 'sigma': 0.014}

    # monthly average parameter shifts from AAPL, (indepedently) conditional on df decrease, mu/sigma increase
    # these are the directions that increase our upper tail risk estimate
    monthly_param_trends = {'df': -0.2210, 'mu': 0.0639 / 100, 'sigma': 0.0875 / 100}

    months_of_stale_calibration = list(range(7))
    percent_backtest_below_1pct = []
    for num_months in tqdm(months_of_stale_calibration, desc="Roughly estimating backtest degradation"):
        new_params = {key: stale_params[key] + monthly_param_trends[key] * num_months for key in stale_params}

        # mimicking backtesting of 2-week returns with a full year of data (non-overlapping windows)
        p_values = stale_calibration_comparison(
            stale_t_params=stale_params, new_t_params=new_params, num_days=10, num_samples_per_trial=25,
            num_trials=10_000, num_paths=10_000, confidence_level=0.95
        )
        percent_backtest_below_1pct.append(sum(np.array(p_values) < 0.01) / len(p_values))

    plt.plot(months_of_stale_calibration, percent_backtest_below_1pct)
    plt.xlabel("Number of months since last calibration")
    plt.ylabel("Percent of backtest $p$-values below 1%")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.ylim(0, plt.ylim()[1])
    plt.title("Example rough extrapolation of backtest degradation\nas parameters drift over time\n"
              rf"Starting parameters: $\nu={stale_params['df']:.2f}$, "
              rf"$\mu={100 * stale_params['mu']:.2f}\%$, $\sigma={100 * stale_params['sigma']:.2f}\%$")
    plt.tight_layout()


if __name__ == "__main__":
    presentation_font_setup()

    for plot_function in [example_t_fit_to_NVDA, example_t_levy_process_profiles,
                          example_exponential_decay_probabilities, example_parameter_stability,
                          example_stale_calibration_degradation]:
        print(f"Running {plot_function.__name__}...")
        plt.close()
        plot_function()
        plt.savefig(f"{plot_function.__name__}.png")
