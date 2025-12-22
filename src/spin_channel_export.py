import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

# Phase Field Parameters
grid_size = (30, 30, 10)
steps = 200
dt = 0.1

# Coupling parameters
bioelectric_coupling = 0.5
phase_diffusion = 1.0
phase_potential_strength = 2.0
voltage_feedback_strength = 0.4

# Initialization
def initialize_fields(shape):
    voltage = np.random.rand(*shape) * 0.1
    spin = np.random.choice([0, 1], size=shape)
    phase = np.random.rand(*shape)  # phase field (Φ)
    return voltage, spin, phase

# Ginzburg-Landau style phase field update
def update_phase_field(phase, voltage, spin):
    lap = laplace(phase)
    local_potential = phase * (1 - phase) * (phase - 0.5)  # triple-well potential
    coupling_term = bioelectric_coupling * (voltage - 0.5) * (2 * spin - 1)
    dphi = -phase_potential_strength * local_potential + phase_diffusion * lap + coupling_term
    return phase + dt * dphi

# Voltage feedback from phase
def update_voltage_field(voltage, phase):
    voltage_change = voltage_feedback_strength * (phase - 0.5)
    voltage += dt * voltage_change
    return np.clip(voltage, 0.0, 1.0)

# Run Simulation
voltage_grid, spin_states, phase_field = initialize_fields(grid_size)

# Store time-lapse data with spin as channel 2
time_lapse_tensor = np.zeros((steps, 3, *grid_size))

for step in range(steps):
    phase_field = update_phase_field(phase_field, voltage_grid, spin_states)
    voltage_grid = update_voltage_field(voltage_grid, phase_field)
    time_lapse_tensor[step, 0] = voltage_grid
    time_lapse_tensor[step, 1] = phase_field
    time_lapse_tensor[step, 2] = spin_states  # Spin is static in this simulation
    if step % 50 == 0:
        print(f"Step {step} complete")

# Export full stack as .npy
np.save("phase_voltage_spin_timelapse.npy", time_lapse_tensor)
print("Exported full-state tensor to phase_voltage_spin_timelapse.npy")
