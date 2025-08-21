import networkx as nx

def get_spin_domains(graph, spin_value=1):
    subgraph = graph.copy()
    for node in list(subgraph.nodes()):
        if subgraph.nodes[node]['spin'] != spin_value:
            subgraph.remove_node(node)
    components = list(nx.connected_components(subgraph))
    return components

def report_largest_spin_domain(graph, spin_value=1):
    domains = get_spin_domains(graph, spin_value)
    if not domains:
        print(f"No spin-{spin_value} domains found.")
        return
    largest = max(domains, key=len)
    print(f"Largest spin-{spin_value} domain size: {len(largest)} nodes")

