import networkx as nx
import torch
from torch_geometric.data import Data

def convert_to_pyg_data(graph):
    node_attrs = []
    for _, data in graph.nodes(data=True):
        voltage = data.get("voltage", 0.0)
        state = data.get("state", 0.0)
        spin = data.get("spin", 0.0)
        node_attrs.append([voltage, state, spin])

    x = torch.tensor(node_attrs, dtype=torch.float)

    edge_index = []
    for u, v in graph.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])  # Make undirected

    node_to_index = {node: i for i, node in enumerate(graph.nodes())}
    edge_index_tensor = torch.tensor(
        [[node_to_index[u], node_to_index[v]] for u, v in graph.edges()] +
        [[node_to_index[v], node_to_index[u]] for u, v in graph.edges()],
        dtype=torch.long
    ).t().contiguous()

    data = Data(x=x, edge_index=edge_index_tensor)
    return data

def export_pyg_data(graph, filename="pyg_graph.pt"):
    data = convert_to_pyg_data(graph)
    torch.save(data, filename)
    print(f"PyTorch Geometric data saved to {filename}")

# Example usage:
# from entangled_spin_grid import create_bioelectric_graph
# graph = create_bioelectric_graph(voltage_grid, cell_states, spin_states)
# export_pyg_data(graph)

