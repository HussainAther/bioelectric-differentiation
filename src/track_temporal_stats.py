def log_stats(step, voltage_grid, spin_grid, cell_state, stats_list):
    avg_voltage = voltage_grid.mean()
    spin_ratio = spin_grid.sum() / spin_grid.size
    differentiated = cell_state.sum()
    stats_list.append({
        "step": step,
        "avg_voltage": avg_voltage,
        "spin_ratio": spin_ratio,
        "differentiated": differentiated
    })

def save_stats_csv(stats_list, filename="simulation_stats.csv"):
    import csv
    keys = stats_list[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(stats_list)

