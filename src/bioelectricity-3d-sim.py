import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

# Simulation Parameters
grid_size = (30, 30, 10)  # 3D Grid (30x30x10 cells)
diffusion_rate = 0.2   # Rate of voltage diffusion
decay_rate = 0.05      # Natural decay of voltage
stimulus_strength = 1.0  # External stimulus strength
steps = 200             # Number of simulation steps
threshold_potential = 0.6  # Threshold for cellular differentiation

def initialize_grid(size):
    return np.random.rand(*size) * 0.1  # Small random voltages

# Initialize voltage grid
voltage_grid = initialize_grid(grid_size)
cell_states = np.zeros(grid_size)  # 0 = undifferentiated, 1 = differentiated

# Function to update the voltage grid
def update_voltage(grid, states):
    new_grid = grid.copy()
    for i in range(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            for k in range(1, grid.shape[2] - 1):
                # Compute Laplacian (reaction-diffusion term)
                laplacian = (
                    grid[i-1, j, k] + grid[i+1, j, k] +
                    grid[i, j-1, k] + grid[i, j+1, k] +
                    grid[i, j, k-1] + grid[i, j, k+1] -
                    6 * grid[i, j, k]
                )
                
                # Update rule: diffusion + decay + external stimulus
                new_grid[i, j, k] += diffusion_rate * laplacian - decay_rate * grid[i, j, k]
                
                # Apply external stimulus to random locations
                if np.random.rand() < 0.01:
                    new_grid[i, j, k] += stimulus_strength
                
                # Cellular differentiation logic
                if new_grid[i, j, k] > threshold_potential:
                    states[i, j, k] = 1  # Mark as differentiated
    
    return np.clip(new_grid, 0, 1), states

# Graph Representation of the Bioelectric Network
def create_bioelectric_graph(grid):
    G = nx.grid_graph(dim=[grid.shape[0], grid.shape[1], grid.shape[2]])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                G.nodes[(i, j, k)]['voltage'] = grid[i, j, k]
                G.nodes[(i, j, k)]['state'] = cell_states[i, j, k]
    return G

bioelectric_graph = create_bioelectric_graph(voltage_grid)

def update_graph(G, grid, states):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                G.nodes[(i, j, k)]['voltage'] = grid[i, j, k]
                G.nodes[(i, j, k)]['state'] = states[i, j, k]
    return G

# Run Simulation
for step in range(steps):
    voltage_grid, cell_states = update_voltage(voltage_grid, cell_states)
    bioelectric_graph = update_graph(bioelectric_graph, voltage_grid, cell_states)

# Visualization: 2D slice of the 3D grid
fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis")
im = ax.imshow(voltage_grid[:, :, 5], cmap=cmap, vmin=0, vmax=1)
plt.colorbar(im, label="Voltage Potential")
plt.title("Mid-Layer Voltage Distribution")
plt.show()

