import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Import CUDA simulation step
from fast_3d_sim import GRID_SIZE, states, spins, voltage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_3d_grid_edges_cuda(depth, height, width):
    """Generates 3D grid graph edge indices directly on GPU memory."""
    num_nodes = depth * height * width
    node_ids = torch.arange(num_nodes, device=DEVICE).reshape(
        depth, height, width
    )

    edges_list = []

    # Edges along depth (X)
    e_depth = torch.stack(
        [node_ids[:-1, :, :].flatten(), node_ids[1:, :, :].flatten()], dim=0
    )
    edges_list.append(e_depth)

    # Edges along height (Y)
    e_height = torch.stack(
        [node_ids[:, :-1, :].flatten(), node_ids[:, 1:, :].flatten()], dim=0
    )
    edges_list.append(e_height)

    # Edges along width (Z)
    e_width = torch.stack(
        [node_ids[:, :, :-1].flatten(), node_ids[:, :, 1:].flatten()], dim=0
    )
    edges_list.append(e_width)

    # Concatenate forward directed edges
    forward_edges = torch.cat(edges_list, dim=1)

    # Make undirected by adding backward edges
    backward_edges = torch.stack([forward_edges[1], forward_edges[0]], dim=0)

    return torch.cat([forward_edges, backward_edges], dim=1)


def export_sim_to_pyg_cuda(voltage_tensor, states_tensor, spins_tensor):
    """Converts 3D PyTorch tensors directly to PyTorch Geometric Data on CUDA without CPU loops."""
    # Reshape node features (voltage, state, spin) to (N, 3) tensor
    v_flat = voltage_tensor.squeeze().reshape(-1, 1)
    st_flat = states_tensor.squeeze().reshape(-1, 1)
    sp_flat = spins_tensor.squeeze().reshape(-1, 1)

    x = torch.cat([v_flat, st_flat, sp_flat], dim=1).float().to(DEVICE)

    d, h, w = voltage_tensor.squeeze().shape
    edge_index = build_3d_grid_edges_cuda(d, h, w)

    return Data(x=x, edge_index=edge_index)


# Graph Attention Network Architecture
class GAT(torch.nn.Module):

    def __init__(self, input_dim=3, hidden_dim=16, output_dim=2, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False
        )

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


if __name__ == "__main__":
    print(f"Exporting 3D Simulation grid to PyG Graph structure on CUDA...")
    data = export_sim_to_pyg_cuda(voltage, states, spins)

    print(
        f"Graph construct complete: {data.x.size(0)} nodes, {data.edge_index.size(1)} edges."
    )
    print(
        f"Training GAT model directly on {torch.cuda.get_device_name(0)}...\n"
    )

    model = GAT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = torch.nn.CrossEntropyLoss()

    target = data.x[:, 2].long()  # Predict spin state from local voltage/state

    start_time = time.time()

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            pred = out.argmax(dim=1)
            acc = pred.eq(target).sum().item() / target.size(0)
            print(
                f"[GAT CUDA] Epoch {epoch:03d} | Loss: {loss.item():.4f} | Spin Pred Accuracy: {acc*100:.2f}%"
            )

    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(
        f"\nEntire pipeline execution time on RTX 4070: {elapsed:.4f} seconds"
    )