import matplotlib.pyplot as plt


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
