import numpy as np
from scipy.stats import binom
from tqdm import trange

from src.simulation import simulate_t_levy_process, tail_risk_estimate


def backtesting_p_value(num_breaches, num_trials, confidence_level=0.95):
    """
    Calculate the p-value for backtesting breaches using a simple binomial test.

    :param num_breaches: Number of breaches observed
    :param num_trials: Total number of trials
    :param confidence_level: Expected probability of not having a breach
    :return: p-value for the observed number of breaches
    """
    num_successes = num_trials - num_breaches
    p_value = binom.cdf(k=num_successes, n=num_trials, p=confidence_level)
    return p_value


def stale_calibration_comparison(stale_t_params, new_t_params, num_days, num_samples_per_trial,
                                 num_trials=10_000, num_paths=10_000, confidence_level=0.95):
    """
    "Backtest" a stale calibration by comparing its tail risk estimate against a newly calibrated t-Levy process.

    :param stale_t_params: t-distribution parameters for the stale calibration (dictionary with keys 'df', 'mu',
    'sigma')
    :param new_t_params: t-distribution parameters for the new calibration (dictionary with keys 'df', 'mu', 'sigma')
    :param num_days: Number of days to simulate in backtesting
    :param num_samples_per_trial: Number of samples to draw in each backtesting trial
    :param num_trials: Number of backtesting trials to run
    :param num_paths: Number of simulation paths to generate for each trial sample
    :param confidence_level: Tail risk percentile to estimate
    :return: List of p-values for all backtesting trails
    """
    # simulate tail risk estimate from stale calibration
    # for stability, we use a much larger number of paths for this estimate than the trial samples
    stale_tail_risk = tail_risk_estimate(
        simulation_paths=simulate_t_levy_process(num_days=num_days, num_paths=num_paths * 100, S0=100,
                                                 df=stale_t_params['df'], mu=stale_t_params['mu'],
                                                 sigma=stale_t_params['sigma']),
        p=confidence_level * 100, both_tails=False, return_profile=False
    )

    p_values = []
    for _ in trange(num_trials, desc="Stale calibration backtesting trials"):
        new_calibration_samples = simulate_t_levy_process(
            num_days=num_days, num_paths=num_samples_per_trial, S0=100,
            df=new_t_params['df'], mu=stale_t_params['mu'], sigma=stale_t_params['sigma']
        )[-1, :]  # just the final time horizon

        num_breaches = np.sum(new_calibration_samples > stale_tail_risk)
        p_values.append(backtesting_p_value(num_breaches, num_samples_per_trial, confidence_level))
    return p_values
