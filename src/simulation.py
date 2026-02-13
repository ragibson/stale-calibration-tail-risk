import numpy as np
from scipy.stats import t


def simulate_t_levy_process(num_days, num_paths, S0, df, mu, sigma):
    """
    Simulate a stock price whose daily log returns are driven by a t-distribution Levy process.

    :param num_days: Number of days to simulate
    :param num_paths: Number of paths to simulate
    :param S0: Initial stock price
    :param df, mu, sigma: Parameters for the t-distribution (degrees of freedom, location, scale)
    :return: Simulated price matrix
    """
    daily_log_returns = t.rvs(df=df, loc=mu, scale=sigma, size=(num_days, num_paths))
    cumulative_log_returns = np.cumsum(daily_log_returns, axis=0)

    # add a t=0 row of zero returns to represent the initial price
    cumulative_log_returns = np.insert(cumulative_log_returns, 0, 0, axis=0)

    price_series = S0 * np.exp(cumulative_log_returns)
    return price_series
