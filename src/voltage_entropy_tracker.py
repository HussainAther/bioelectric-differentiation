import numpy as np
import pandas as pd
from scipy.stats import entropy

# Parameters
grid_size = (30, 30, 10)
steps = 200
bins = 20

# Entropy of a 3D field
def compute_entropy(field, bins=20):
    hist, _ = np.histogram(field.flatten(), bins=bins, density=True)
    hist += 1e-9  # avoid log(0)
    return entropy(hist, base=2)

# Track metrics over time
def track_entropy_over_time(tensor, save_csv=True):
    voltage_series = []
    phase_series = []
    spin_series = []

    for t in range(tensor.shape[0]):
        voltage = tensor[t, 0]
        phase = tensor[t, 1]
        spin = tensor[t, 2]

        voltage_series.append([
            t,
            compute_entropy(voltage, bins),
            np.mean(voltage),
            np.var(voltage)
        ])

        phase_series.append([
            t,
            compute_entropy(phase, bins),
            np.mean(phase),
            np.var(phase)
        ])

        spin_series.append([
            t,
            compute_entropy(spin.astype(float), bins),
            np.mean(spin),
            np.var(spin)
        ])

    df_voltage = pd.DataFrame(voltage_series, columns=["step", "entropy", "mean", "variance"])
    df_phase = pd.DataFrame(phase_series, columns=["step", "entropy", "mean", "variance"])
    df_spin = pd.DataFrame(spin_series, columns=["step", "entropy", "mean", "variance"])

    if save_csv:
        df_voltage.to_csv("voltage_entropy_log.csv", index=False)
        df_phase.to_csv("phase_entropy_log.csv", index=False)
        df_spin.to_csv("spin_entropy_log.csv", index=False)

    return df_voltage, df_phase, df_spin

# Example usage
if __name__ == "__main__":
    data = np.load("phase_voltage_spin_ising_timelapse.npy")
    track_entropy_over_time(data)
    print("Entropy logs saved.")

