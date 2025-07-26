import argparse
from entangled_spin_grid import initialize_grid, update_grid, create_bioelectric_graph, export_graph_data

def run_sim(grid_size, steps, output):
    voltage, states, spins = initialize_grid(grid_size)
    for _ in range(steps):
        voltage, states, spins = update_grid(voltage, states, spins)
    graph = create_bioelectric_graph(voltage, states, spins)
    export_graph_data(graph, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bioelectric Spin Simulation")
    parser.add_argument("--size", type=int, nargs=3, default=[30, 30, 10], help="Grid size (X Y Z)")
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps")
    parser.add_argument("--output", type=str, default="bioelectric_graph.gexf", help="Output file for graph data")
    args = parser.parse_args()

    run_sim(tuple(args.size), args.steps, args.output)

