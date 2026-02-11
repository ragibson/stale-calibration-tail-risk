import numpy as np
from scipy.optimize import minimize
from scipy.stats import t


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

    # simple unweighted estimates for initial optimization guess
    if not initial_loc:
        initial_loc = np.median(x)  # median instead of mean for robustness
    if not initial_scale:
        # using Var(x) = df / (df - 2) * scale^2
        initial_scale = np.sqrt(np.var(x) * (initial_df - 2) / initial_df)

    result = minimize(
        lambda params: -t_fp_log_likelihood(x, df=params[0], weights=weights, loc=params[1], scale=params[2]),
        x0=[initial_df, initial_loc, initial_scale],
        bounds=[df_bounds, (None, None), scale_bounds]
    )
    return result.x if result.success else None
