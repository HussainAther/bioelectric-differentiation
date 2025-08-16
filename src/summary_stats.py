import networkx as nx

def print_summary(graph):
    voltages = [data["voltage"] for _, data in graph.nodes(data=True)]
    spins = [data["spin"] for _, data in graph.nodes(data=True)]
    states = [data["state"] for _, data in graph.nodes(data=True)]

    print(f"Total nodes: {len(voltages)}")
    print(f"Avg voltage: {sum(voltages) / len(voltages):.4f}")
    print(f"Spin |0⟩ count: {spins.count(0)}")
    print(f"Spin |1⟩ count: {spins.count(1)}")
    print(f"Differentiated cells: {int(sum(states))}")

