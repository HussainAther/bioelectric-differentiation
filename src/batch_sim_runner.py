import os
import random
from entangled_spin_grid import initialize_grid, update_grid, create_bioelectric_graph
from gnn_data_export import export_pyg_data

def run_batch(num_runs=5, grid_size=(30, 30, 10), steps=100, out_dir="batch_outputs"):
    os.makedirs(out_dir, exist_ok=True)

    for seed in range(num_runs):
        random.seed(seed)
        voltage, states, spins = initialize_grid(grid_size)
        for _ in range(steps):
            voltage, states, spins = update_grid(voltage, states, spins)
        graph = create_bioelectric_graph(voltage, states, spins)
        export_pyg_data(graph, filename=os.path.join(out_dir, f"sim_{seed}.pt"))

if __name__ == "__main__":
    run_batch()

