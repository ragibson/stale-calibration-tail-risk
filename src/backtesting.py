from scipy.stats import binom


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
