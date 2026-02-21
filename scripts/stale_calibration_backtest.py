import numpy as np

from src.backtesting import stale_calibration_comparison

if __name__ == "__main__":
    # df and sigma taken from AAPL's COVID crash, changes from 3/9/2020 to 4/14/2020
    stale_params = {'df': 2.9533, 'mu': 0.0, 'sigma': 0.010604}
    new_params = {'df': 2.4844, 'mu': 0.0, 'sigma': 0.010894}

    # mimicking backtesting of 2-week returns with a full year of data (non-overlapping windows)
    p_values = stale_calibration_comparison(
        stale_t_params=stale_params, new_t_params=new_params, num_days=10, num_samples_per_trial=25,
        num_trials=100_000, num_paths=10_000, confidence_level=0.95
    )
    for threshold in [0.01, 0.05]:
        num_below_threshold = sum(np.array(p_values) < threshold)
        print(f"Percent of p-values below {threshold:.0%}: {num_below_threshold / len(p_values):.1%}")
