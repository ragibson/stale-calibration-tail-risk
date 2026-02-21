import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import t, norm


def plot_profile_and_recent_price(historical_prices, calibration_date, profiles, title,
                                  recent_price_length=252 // 12,
                                  profile_labels=("Low Profile (5%)", "High Profile (95%)"),
                                  hide_prices_after_calibration=False):
    """
    Plots historical prices along with low and high profile projections from the calibration date.

    :param historical_prices: Series of historical prices indexed by date
    :param calibration_date: Calibration date to plot profiles from
    :param profiles: Tuple of (low_profile, high_profile) price projections
    :param title: Plot title
    :param recent_price_length: Number of recent prices to plot for context
    :param profile_labels: Labels to use for the low and high profiles in the legend
    :param hide_prices_after_calibration: If True, only plot historical prices up to the calibration date
    """
    low_profile, high_profile = profiles

    # trim prices to a window around the calibration date for visualization purposes
    calibration_date_index = historical_prices.index.get_loc(calibration_date)
    if hide_prices_after_calibration:
        # further trim any prices after the calibration date
        dates_to_plot = historical_prices.index[calibration_date_index - recent_price_length:calibration_date_index + 1]
    else:
        dates_to_plot = historical_prices.index[calibration_date_index - recent_price_length
                                                :calibration_date_index + len(low_profile)]

    plt.plot(dates_to_plot, historical_prices[dates_to_plot], label="Historical Price")

    profile_dates = historical_prices.index[calibration_date_index:calibration_date_index + len(low_profile)]
    plt.plot(profile_dates, high_profile, label=profile_labels[1], linestyle='--')
    plt.plot(profile_dates, low_profile, label=profile_labels[0], linestyle='--')

    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    plt.ylabel("Price")
    plt.xlabel("Date")
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
             label="$t$-distribution fit")

    # additionally fit normal distribution for comparison
    normal_params = norm.fit(data)
    plt.plot(plot_xs, norm.pdf(plot_xs, loc=normal_params[0], scale=normal_params[1]),
             label="Normal distribution fit")

    plt.title(title)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlabel(x_label)
    plt.ylabel("Distribution density")
    plt.legend()
    plt.tight_layout()


def plot_parameter_stability_and_log_price(close_prices, calibration_dates, calibrated_params, figsize=None,
                                           params_to_plot=('df', 'sigma'), param_latex=(r'$\nu$', r'$\sigma$'),
                                           plot_titles=(r'Rolling $\nu$ calibration', r"Rolling $\sigma$ calibration")):
    """
    Plots the stability of calibrated parameters over time with an overlay of the historical log-price.

    :param close_prices: Close prices to plot (Series with a date index)
    :param calibration_dates: Calibration dates corresponding to the list of calibrated parameters
    :param calibrated_params: Calibrated parameters (list of dicts with 'df', 'mu', 'sigma' keys)
    :param figsize: Figure size for full matplotlib plot
    :param params_to_plot: Tuple of parameters to include in the plot
    :param param_latex: LaTeX-formatted labels for each parameter
    :param plot_titles: Titles to use for each parameter subplot
    """
    fig, ax = plt.subplots(nrows=len(params_to_plot), figsize=figsize)
    first_calibration_date = calibration_dates[0]

    for i, param_name in enumerate(params_to_plot):
        param_values = [params[param_name] for params in calibrated_params]
        ln1 = ax[i].plot(calibration_dates, param_values, label=f"Calibrated {param_latex[i]}")

        ax_price = ax[i].twinx()
        ln2 = ax_price.plot(close_prices.index[close_prices.index >= first_calibration_date],
                            np.log(close_prices[close_prices.index >= first_calibration_date]),
                            alpha=0.5, color='black', label="Historical log-price")

        ax[i].set_xlabel("Date")
        ax[i].set_ylabel(param_latex[i])
        ax_price.set_ylabel("Historical log-price")
        ax[i].set_title(plot_titles[i])

        # combining legends from both axes into one
        both_lines = ln1 + ln2
        ax[i].legend(both_lines, [l.get_label() for l in both_lines])

        if param_name != 'df':
            ax[i].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
