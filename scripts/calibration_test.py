import numpy as np
from scipy.stats import t

from src.calibration import calibrate_t_fp

if __name__ == "__main__":
    for _ in range(5):
        data = t.rvs(df=3.5, loc=1.25, scale=0.25, size=252)
        params = calibrate_t_fp(data, weights=np.ones_like(data))
        print(params)
