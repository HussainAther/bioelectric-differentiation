import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import os

# Sample GCN Model for Classification (spin prediction)
def build_gcn_model(input_dim, hidden_dim=32, output_dim=2):
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    return GCN()

# Load all PyG graph files from a folder
def load_graph_dataset(folder_path):
    dataset = []
    for file in os.listdir(folder_path):
        if file.endswith(".pt"):
            data = torch.load(os.path.join(folder_path, file))
            spin_labels = data.x[:, 2].long()  # Treat spin as target
            data.y = spin_labels
            dataset.append(data)
    return dataset

def train(model, dataset, task='classification', epochs=100, lr=0.01):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_nodes = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            target = data.y
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total_nodes += target.size(0)
        acc = correct / total_nodes
        print(f"Epoch {epoch+1:03d}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# Example usage
if __name__ == '__main__':
    dataset = load_graph_dataset("batch_outputs")  # Folder with *.pt graphs
    model = build_gcn_model(input_dim=3, output_dim=2)  # Spin classification
    train(model, dataset, task='classification')

