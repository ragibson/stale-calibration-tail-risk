import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t

from src.data_loading import truncate_series_to_date_range


def exponential_decay_probabilities(tau_hl, t_bar):
    """
    Returns exponential decay probabilities for a given half-life and time series length.

    For example, tau_hl = 1 and t_bar = 4 gives [0.067, 0.133, 0.267, 0.533].

    :param tau_hl: Half-life of the exponential decay in days
    :param t_bar: Length of desired time series (number of observations)
    """
    weights = np.exp(-(np.log(2) / tau_hl) * abs(t_bar - 1 - np.arange(0, t_bar)))
    return weights / sum(weights)  # normalize to sum to 1


def t_fp_log_likelihood(x, df, weights, loc=0, scale=1):
    """Calculate weighted variant of t-distribution log-likelihood."""
    weights /= np.sum(weights)  # force normalization of sum(weights) = 1
    return np.sum(weights * t.logpdf(x, df=df, loc=loc, scale=scale))


def calibrate_t_fp(x, weights, initial_df=5, initial_loc=None, initial_scale=None,
                   df_bounds=(2.0, None), scale_bounds=(1e-6, None)):
    """
    Calibrate t-distribution parameters using maximum likelihood with flexible probabilities estimates.

    Note that the default bounds for df ensure the variance is finite.
    """
    if np.isnan(x).any():
        raise ValueError("Calibration data contains NaN values!")

    # simple unweighted estimates for initial optimization guess
    if not initial_loc:
        initial_loc = np.median(x)  # median instead of mean for robustness
    if not initial_scale:
        # using Var(x) = df / (df - 2) * scale^2
        initial_scale = np.sqrt(np.var(x) * (initial_df - 2) / initial_df)
        initial_scale = np.clip(initial_scale, *scale_bounds)

    result = minimize(
        lambda params: -t_fp_log_likelihood(x, df=params[0], weights=weights, loc=params[1], scale=params[2]),
        x0=[initial_df, initial_loc, initial_scale],
        bounds=[df_bounds, (None, None), scale_bounds],
        method='Powell',
        tol=1e-8, options={'maxiter': 10_000}
    )
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    return dict(zip(['df', 'mu', 'sigma'], result.x))


def calibrate_t_levy_process(daily_log_returns, end_date, num_years, tau_hl_df=252, tau_hl_sigma=252 // 4):
    """
    Calibrate a t-distribution Levy process using flexible probabilities with exponential decay.

    :param daily_log_returns: Dataframe of daily log returns with a DateTime index
    :param end_date: Final date to include in the calibration period
    :param num_years: Number of years to include in the calibration period
    :param tau_hl_df: Half-life (in days) for the exponential decay of the degrees of freedom parameter
    :param tau_hl_sigma: Half-life (in days) for the exponential decay of the scale parameter
    :return: Calibrated degrees of freedom, mu (location), and sigma (scale) parameters of the t-distribution
    """
    calibration_period = truncate_series_to_date_range(daily_log_returns, end_date, num_years)
    calibration_values = calibration_period.values.flatten()

    # first, calibrate the degrees of freedom
    if tau_hl_df is not None:
        df_weights = exponential_decay_probabilities(tau_hl=tau_hl_df, t_bar=len(calibration_period))
    else:
        df_weights = 1.0 / np.ones(len(calibration_period))
    df_param = calibrate_t_fp(calibration_values, weights=df_weights)["df"]

    # then, calibrate the location and scale parameters
    if tau_hl_sigma is not None:
        sigma_weights = exponential_decay_probabilities(tau_hl=tau_hl_sigma, t_bar=len(calibration_period))
    else:
        sigma_weights = 1.0 / np.ones(len(calibration_period))
    params = calibrate_t_fp(calibration_values, weights=sigma_weights, initial_df=df_param,
                            df_bounds=(df_param, df_param))

    return params


def first_available_calibration_date(data, num_years):
    """Determine the first date a calibration can be performed given a time series and number of years."""
    start_date = data.index[0] + pd.DateOffset(years=num_years)
    return start_date if start_date <= data.index[-1] else None
