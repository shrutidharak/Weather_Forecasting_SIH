import torch
from typing import List
import torch.optim as optim
import sys
import os

try:
    from .gnn_tracker import LightweightTrackingGNN, GNNTracker
except ImportError:
    from gnn_tracker import LightweightTrackingGNN, GNNTracker

def train_gnn(model: LightweightTrackingGNN, epochs: int = 200):
    """
    Very simple synthetic training loop to force the GNN to learn spatial/feature similarity and momentum.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Create a synthetic batch of nodes at t
        # [lat, lon, area, intensity, vel_lat, vel_lon]
        N = 10
        nodes_t = torch.rand(N, 6)
        
        # Give them realistic velocity bounds (-0.1 to 0.1)
        nodes_t[:, 4] = (torch.rand(N) - 0.5) * 0.2
        nodes_t[:, 5] = (torch.rand(N) - 0.5) * 0.2
        
        # Simulate nodes at t+1 by moving them by their velocity, adding slight noise
        expected_pos_lat = nodes_t[:, 0] + nodes_t[:, 4]
        expected_pos_lon = nodes_t[:, 1] + nodes_t[:, 5]
        
        nodes_t1 = nodes_t.clone()
        nodes_t1[:, 0] = expected_pos_lat + torch.randn(N) * 0.01
        nodes_t1[:, 1] = expected_pos_lon + torch.randn(N) * 0.01
        
        # At t1, they don't have velocity computed yet since they just arrived
        nodes_t1[:, 4] = 0.0
        nodes_t1[:, 5] = 0.0
        
        # The correct assignment is identity (before shuffle)
        # We will just shuffle nodes_t1
        indices = torch.randperm(N)
        nodes_t1 = nodes_t1[indices]
        
        # Target matrix (N, N) where target[i, j] = 1 if j == indices[i]
        target = torch.zeros(N, N)
        for i in range(N):
            target[i, (indices == i).nonzero(as_tuple=True)[0]] = 1.0
            
        scores = model(nodes_t, nodes_t1)
        loss = criterion(scores, target)
        
        loss.backward()
        optimizer.step()
        
    return model

if __name__ == "__main__":
    model = LightweightTrackingGNN()
    print("Training GNN on synthetic proximity task...")
    model = train_gnn(model, epochs=300)
    print("Training complete. Saving model.")
    torch.save(model.state_dict(), "gnn_tracker.pth")
