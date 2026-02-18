import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import t


def plot_profile_and_recent_price(historical_prices, calibration_date, profiles,
                                  recent_price_length=252 // 12):
    """
    Plots historical prices along with low and high profile projections from the calibration date.

    :param historical_prices: Series of historical prices indexed by date
    :param calibration_date: Calibration date to plot profiles from
    :param profiles: Tuple of (low_profile, high_profile) price projections
    :param recent_price_length: Number of recent prices to plot for context
    """
    low_profile, high_profile = profiles

    # trim prices to a window around the calibration date for visualization purposes
    calibration_date_index = historical_prices.index.get_loc(calibration_date)
    dates_to_plot = historical_prices.index[calibration_date_index - recent_price_length
                                            :calibration_date_index + len(low_profile)]
    plt.plot(dates_to_plot, historical_prices[dates_to_plot], label="Historical Price")

    profile_dates = historical_prices.index[calibration_date_index:calibration_date_index + len(low_profile)]

    plt.plot(profile_dates, low_profile, label="Low Profile", linestyle='--')
    plt.plot(profile_dates, high_profile, label="High Profile", linestyle='--')
    plt.legend()
    plt.tight_layout()


def plot_histogram_and_t_fit(data, t_params, title, plot_label, x_label):
    """
    Plots a histogram of specified data with an overlaid t-distribution fit based on the provided parameters.

    Here, the x-axis is formatted as a percentage.

    :param data: Histogram data to plot
    :param t_params: t-distribution parameters (dictionary with keys 'df', 'mu', 'sigma')
    :param title: Plot title
    :param plot_label: Label to use for histogram legend
    :param x_label: Label to use for x-axis
    """
    plt.hist(data, bins=100, density=True, label=plot_label)

    # using histogram plot bounds to plot the t-distribution PDF
    plot_xs = np.linspace(*plt.xlim(), 1_000)
    plt.plot(plot_xs, t.pdf(plot_xs, df=t_params['df'], loc=t_params['mu'], scale=t_params['sigma']),
             label="t-distribution fit")

    plt.title(title)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlabel(x_label)
    plt.ylabel("Distribution density")
    plt.legend()
    plt.tight_layout()
