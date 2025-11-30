import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data_utils import load_graph_data

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def train_gat(data, epochs=100, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(input_dim=3, hidden_dim=16, output_dim=2).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        target = data.x[:, 2].long()
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = pred.eq(target).sum().item() / target.size(0)
        print(f"[GAT] Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {acc:.4f}")

if __name__ == "__main__":
    data = load_graph_data("data/graph_data.pt")
    train_gat(data)

