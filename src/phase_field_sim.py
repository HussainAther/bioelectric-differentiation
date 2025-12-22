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

# Run Simulation
voltage_grid, spin_states, phase_field = initialize_fields(grid_size)

for step in range(steps):
    phase_field = update_phase_field(phase_field, voltage_grid, spin_states)
    if step % 50 == 0:
        print(f"Step {step} complete")

# Visualize central Z slice of phase field
mid_z = grid_size[2] // 2
plt.imshow(phase_field[:, :, mid_z], cmap='plasma')
plt.title("Phase Field Slice at Z Midplane")
plt.colorbar(label='Phase (Φ)')
plt.tight_layout()
plt.show()

