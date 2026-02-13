import matplotlib.pyplot as plt
import numpy as np

from src.simulation import simulate_t_levy_process

if __name__ == "__main__":
    simulation_paths = simulate_t_levy_process(
        num_days=252 // 12, num_paths=10_000, S0=100, df=4, mu=0.0015, sigma=0.015
    )

    p95 = np.percentile(simulation_paths, 95, axis=1)
    mean = np.mean(simulation_paths, axis=1)
    p05 = np.percentile(simulation_paths, 5, axis=1)

    ts = np.arange(simulation_paths.shape[0])
    plt.plot(ts, p95, label="95th Percentile")
    plt.plot(ts, mean, label="Mean")
    plt.plot(ts, p05, label="5th Percentile")
    plt.legend()
    plt.tight_layout()
    plt.show()
