import networkx as nx
import matplotlib.pyplot as plt

def plot_spin_distribution(graph):
    spins = [data['spin'] for _, data in graph.nodes(data=True)]
    plt.hist(spins, bins=[-0.5, 0.5, 1.5], rwidth=0.8, color='purple')
    plt.xticks([0, 1], labels=['|0⟩', '|1⟩'])
    plt.xlabel('Spin State')
    plt.ylabel('Count')
    plt.title('Spin State Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_coherence(graph):
    spins = nx.get_node_attributes(graph, 'spin')
    values = list(spins.values())
    return sum(values) / len(values)

def average_voltage(graph):
    voltages = nx.get_node_attributes(graph, 'voltage')
    return sum(voltages.values()) / len(voltages)

