import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

# Parameters
grid_size = (30, 30, 10)
steps = 200
dt = 0.1

# Coupling parameters
bioelectric_coupling = 0.5
phase_diffusion = 1.0
phase_potential_strength = 2.0
voltage_feedback_strength = 0.4
ising_temperature = 0.1  # Lower = less likely spin flips

# Initialization
def initialize_fields(shape):
    voltage = np.random.rand(*shape) * 0.1
    spin = np.random.choice([0, 1], size=shape)
    phase = np.random.rand(*shape)
    return voltage, spin, phase

# Update phase field
def update_phase_field(phase, voltage, spin):
    lap = laplace(phase)
    local_potential = phase * (1 - phase) * (phase - 0.5)
    coupling_term = bioelectric_coupling * (voltage - 0.5) * (2 * spin - 1)
    dphi = -phase_potential_strength * local_potential + phase_diffusion * lap + coupling_term
    return phase + dt * dphi

# Update voltage based on phase
def update_voltage_field(voltage, phase):
    voltage_change = voltage_feedback_strength * (phase - 0.5)
    voltage += dt * voltage_change
    return np.clip(voltage, 0.0, 1.0)

# Ising-style spin update
def update_spin_states(spins, voltage, phase):
    updated_spins = spins.copy()
    shape = spins.shape
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            for k in range(1, shape[2]-1):
                neighbors = [
                    spins[i-1, j, k], spins[i+1, j, k],
                    spins[i, j-1, k], spins[i, j+1, k],
                    spins[i, j, k-1], spins[i, j, k+1]
                ]
                local_field = sum([1 if s == 1 else -1 for s in neighbors])
                energy_diff = -2 * (2 * spins[i, j, k] - 1) * local_field + (phase[i, j, k] - 0.5) * 2
                prob = 1 / (1 + np.exp(energy_diff / ising_temperature))
                if np.random.rand() < prob:
                    updated_spins[i, j, k] = 1 - spins[i, j, k]
    return updated_spins

# Run Simulation
voltage_grid, spin_states, phase_field = initialize_fields(grid_size)
time_lapse_tensor = np.zeros((steps, 3, *grid_size))

for step in range(steps):
    phase_field = update_phase_field(phase_field, voltage_grid, spin_states)
    voltage_grid = update_voltage_field(voltage_grid, phase_field)
    spin_states = update_spin_states(spin_states, voltage_grid, phase_field)
    time_lapse_tensor[step, 0] = voltage_grid
    time_lapse_tensor[step, 1] = phase_field
    time_lapse_tensor[step, 2] = spin_states
    if step % 50 == 0:
        print(f"Step {step} complete")

# Export full tensor
np.save("phase_voltage_spin_ising_timelapse.npy", time_lapse_tensor)
print("Saved: phase_voltage_spin_ising_timelapse.npy")

