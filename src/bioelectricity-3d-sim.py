import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

# Simulation Parameters
grid_size = (30, 30, 10)
diffusion_rate = 0.2
decay_rate = 0.05
stimulus_strength = 1.0
steps = 100
threshold_potential = 0.6
kT = 0.05  # Thermal energy for QTM probability (bioelectric analogy)
coupling_strength = 0.1  # Strength of spin coupling with neighbors

# Initialize voltage grid, cell states, and spin states
def initialize_grid(size):
    voltage = np.random.rand(*size) * 0.1
    cell_state = np.zeros(size)  # 0 = undifferentiated, 1 = differentiated
    spin_state = np.random.choice([0, 1], size=size)  # |0> or |1> spin state
    return voltage, cell_state, spin_state

voltage_grid, cell_states, spin_states = initialize_grid(grid_size)

# Bistability and QTM simulation
def qtm_flip(prob):
    return np.random.rand() < prob

# Spin coupling influence
def neighbor_spin_coupling(i, j, k, spins):
    neighbors = [
        spins[i-1, j, k], spins[i+1, j, k],
        spins[i, j-1, k], spins[i, j+1, k],
        spins[i, j, k-1], spins[i, j, k+1]
    ]
    return np.mean(neighbors)

# Update voltage, differentiation, and quantum state
def update_grid(voltage, states, spins):
    new_voltage = voltage.copy()
    for i in range(1, voltage.shape[0] - 1):
        for j in range(1, voltage.shape[1] - 1):
            for k in range(1, voltage.shape[2] - 1):
                laplacian = (
                    voltage[i-1, j, k] + voltage[i+1, j, k] +
                    voltage[i, j-1, k] + voltage[i, j+1, k] +
                    voltage[i, j, k-1] + voltage[i, j, k+1] -
                    6 * voltage[i, j, k]
                )
                new_voltage[i, j, k] += diffusion_rate * laplacian - decay_rate * voltage[i, j, k]

                if np.random.rand() < 0.01:
                    new_voltage[i, j, k] += stimulus_strength

                # Bistability: differentiate or revert depending on potential
                if new_voltage[i, j, k] > threshold_potential:
                    states[i, j, k] = 1
                elif new_voltage[i, j, k] < threshold_potential / 2:
                    states[i, j, k] = 0

                # QTM + Neighbor Spin Coupling
                neighbor_avg_spin = neighbor_spin_coupling(i, j, k, spins)
                coupling_effect = coupling_strength * (1 - abs(spins[i, j, k] - neighbor_avg_spin))
                energy_barrier = abs(0.5 - new_voltage[i, j, k]) - coupling_effect
                tunneling_prob = np.exp(-max(energy_barrier, 0) / kT)
                if qtm_flip(tunneling_prob):
                    spins[i, j, k] = 1 - spins[i, j, k]  # Flip spin

    return np.clip(new_voltage, 0, 1), states, spins

# Create 3D graph representation
def create_bioelectric_graph(grid, states, spins):
    G = nx.grid_graph(dim=[grid.shape[0], grid.shape[1], grid.shape[2]])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                G.nodes[(i, j, k)]['voltage'] = grid[i, j, k]
                G.nodes[(i, j, k)]['state'] = states[i, j, k]
                G.nodes[(i, j, k)]['spin'] = spins[i, j, k]
    return G

# Run Simulation
for step in range(steps):
    voltage_grid, cell_states, spin_states = update_grid(voltage_grid, cell_states, spin_states)

bioelectric_graph = create_bioelectric_graph(voltage_grid, cell_states, spin_states)

# Visualize differentiated cells with spin states
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
diff_cells = np.array(np.where(cell_states == 1)).T
colors = ['blue' if spin_states[tuple(cell)] == 0 else 'red' for cell in diff_cells]
ax.scatter(diff_cells[:, 0], diff_cells[:, 1], diff_cells[:, 2], c=colors, marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Bioelectric Differentiation with Spin Coupling')
plt.show()

