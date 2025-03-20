import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

# Simulation Parameters
grid_size = (50, 50)  # Grid dimensions
diffusion_rate = 0.2   # Rate of voltage diffusion
decay_rate = 0.05      # Natural decay of voltage
stimulus_strength = 1.0  # External stimulus strength
steps = 200             # Number of simulation steps

def initialize_grid(size):
    return np.random.rand(*size) * 0.1  # Small random voltages

# Initialize voltage grid
voltage_grid = initialize_grid(grid_size)

# Function to update the voltage grid
def update_voltage(grid):
    new_grid = grid.copy()
    for i in range(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            # Compute Laplacian (reaction-diffusion term)
            laplacian = (
                grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1]
                - 4 * grid[i, j]
            )
            
            # Update rule: diffusion + decay + external stimulus
            new_grid[i, j] += diffusion_rate * laplacian - decay_rate * grid[i, j]
            
            # Apply external stimulus to random locations
            if np.random.rand() < 0.01:
                new_grid[i, j] += stimulus_strength
    
    return np.clip(new_grid, 0, 1)  # Keep voltages between 0 and 1

# Graph Representation of the Bioelectric Network
def create_bioelectric_graph(grid):
    G = nx.grid_2d_graph(grid.shape[0], grid.shape[1])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            G.nodes[(i, j)]['voltage'] = grid[i, j]
    return G

bioelectric_graph = create_bioelectric_graph(voltage_grid)

def update_graph(G, grid):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            G.nodes[(i, j)]['voltage'] = grid[i, j]
    return G

# Animate the simulation
fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis")
im = ax.imshow(voltage_grid, cmap=cmap, vmin=0, vmax=1)

def animate(frame):
    global voltage_grid, bioelectric_graph
    voltage_grid = update_voltage(voltage_grid)
    bioelectric_graph = update_graph(bioelectric_graph, voltage_grid)
    im.set_array(voltage_grid)
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)
plt.show()

