import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data

from data_utils import load_graph_data

class MPNN(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim):
        super(MPNN, self).__init__()
        edge_nn = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, input_dim * hidden_dim)
        )
        self.conv = NNConv(input_dim, hidden_dim, edge_nn, aggr='mean')
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv(x, edge_index, edge_attr))
        return self.lin(x)

def train_mpnn(data, epochs=100, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MPNN(input_dim=3, edge_dim=1, hidden_dim=16, output_dim=2).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        target = data.x[:, 2].long()
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = pred.eq(target).sum().item() / target.size(0)
        print(f"[MPNN] Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {acc:.4f}")

if __name__ == "__main__":
    data = load_graph_data("data/graph_data.pt")
    # You must have .edge_attr set correctly as 1D edge features
    train_mpnn(data)

