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


def tail_risk_estimate(simulation_paths, p, return_profile=False, both_tails=False):
    """
    Estimate the p-th percentile tail risk from simulated price paths.

    :param simulation_paths: Matrix of simulated price paths (time steps, paths)
    :param p: Percentile to estimate
    :param return_profile: if True, return the percentile profile for all simulated time horizons
    :param both_tails: if True, return both the left (100-p)-th and right p-th percentiles
    """
    if both_tails:
        return tuple(tail_risk_estimate(simulation_paths, p, return_profile=return_profile,
                                        both_tails=False) for p in (100 - p, p))

    if return_profile:
        return np.percentile(simulation_paths, p, axis=1)

    # only returning the final time horizon
    return np.percentile(simulation_paths[-1, :], p)
