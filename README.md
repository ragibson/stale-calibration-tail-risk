# stale-calibration-tail-risk

This experiment evaluates the impact of stale model calibrations on tail risk estimates from the perspective of
statistical backtesting.

Here, daily log stock returns are modeled as a t-distribution Levy process. Model parameters are estimated via maximum
likelihood using exponentially decaying flexible probabilities.

The backtesting impact is then measured by how the p-value distribution of a binomial backtest shifts when using a stale
calibration instead of a freshly updated one.

* [presentation/](presentation/) contains slides for the presentation itself
* [src/](src/) contains the Python implementation
* [scripts/](scripts/)  contains scripts that demonstrate model usage and generate the presentation content
