import networkx as nx
import csv

def export_voltage_gradients(graph, filename="voltage_gradients.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_u", "node_v", "voltage_u", "voltage_v", "gradient"])
        for u, v in graph.edges():
            v_u = graph.nodes[u].get("voltage", 0.0)
            v_v = graph.nodes[v].get("voltage", 0.0)
            gradient = abs(v_u - v_v)
            writer.writerow([u, v, v_u, v_v, gradient])

