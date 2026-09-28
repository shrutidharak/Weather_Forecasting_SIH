import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

try:
    from .baseline_detector import DetectedObject
    from .tracker import TrackedEvent
except ImportError:
    from baseline_detector import DetectedObject
    from tracker import TrackedEvent
    
import uuid

class LightweightTrackingGNN(nn.Module):
    def __init__(self, node_dim: int = 6, hidden_dim: int = 16):
        super().__init__()
        # Node features: [lat, lon, area, intensity, vel_lat, vel_lon]
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, nodes_t, nodes_t1):
        emb_t = self.node_mlp(nodes_t)   # (N, H)
        emb_t1 = self.node_mlp(nodes_t1) # (M, H)
        
        N, H = emb_t.shape
        M, _ = emb_t1.shape
        
        if N == 0 or M == 0:
            return torch.zeros((N, M))
            
        emb_t_expanded = emb_t.unsqueeze(1).expand(N, M, H)
        emb_t1_expanded = emb_t1.unsqueeze(0).expand(N, M, H)
        
        pair_features = torch.cat([emb_t_expanded, emb_t1_expanded], dim=-1)
        scores = self.edge_scorer(pair_features).squeeze(-1)
        return scores

class GNNTracker:
    def __init__(self, model_path: str = None):
        self.model = LightweightTrackingGNN()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.active_tracks: List[TrackedEvent] = []
        self.finished_tracks: List[TrackedEvent] = []
        
    def _track_to_tensor(self, track: TrackedEvent) -> torch.Tensor:
        obj = track.history[-1]
        vel_lat, vel_lon = 0.0, 0.0
        if len(track.history) > 1:
            prev_obj = track.history[-2]
            vel_lat = obj.centroid_lat - prev_obj.centroid_lat
            vel_lon = obj.centroid_lon - prev_obj.centroid_lon
            
        return torch.tensor([
            obj.centroid_lat / 32.0, 
            obj.centroid_lon / 32.0, 
            min(obj.area / 100.0, 1.0), 
            min(obj.intensity / 20.0, 1.0),
            vel_lat,
            vel_lon
        ], dtype=torch.float32)
        
    def _obj_to_tensor(self, obj: DetectedObject) -> torch.Tensor:
        return torch.tensor([
            obj.centroid_lat / 32.0, 
            obj.centroid_lon / 32.0, 
            min(obj.area / 100.0, 1.0), 
            min(obj.intensity / 20.0, 1.0),
            0.0, # Unknown initial velocity
            0.0
        ], dtype=torch.float32)

    def update(self, detected_objects: List[DetectedObject], time_step: int):
        if not self.active_tracks:
            for obj in detected_objects:
                self.active_tracks.append(TrackedEvent(str(uuid.uuid4()), time_step, time_step, [obj]))
            return
            
        if not detected_objects:
            self.finished_tracks.extend(self.active_tracks)
            self.active_tracks = []
            return
            
        # Convert to tensors
        nodes_t = torch.stack([self._track_to_tensor(t) for t in self.active_tracks])
        nodes_t1 = torch.stack([self._obj_to_tensor(obj) for obj in detected_objects])
        
        # Get affinity matrix
        with torch.no_grad():
            scores = self.model(nodes_t, nodes_t1).numpy()
            
        # We want to maximize score, linear_sum_assignment minimizes cost
        cost_matrix = 1.0 - scores
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assigned_tracks = set()
        assigned_objs = set()
        
        next_active = []
        
        for r, c in zip(row_ind, col_ind):
            # Only associate if the model gives it > 0.5 probability (cost < 0.5)
            if cost_matrix[r, c] < 0.5:
                track = self.active_tracks[r]
                obj = detected_objects[c]
                track.history.append(obj)
                track.end_time = time_step
                next_active.append(track)
                assigned_tracks.add(r)
                assigned_objs.add(c)
                
        # Handle unassigned tracks (die)
        for i, track in enumerate(self.active_tracks):
            if i not in assigned_tracks:
                self.finished_tracks.append(track)
                
        # Handle unassigned objects (new tracks)
        for j, obj in enumerate(detected_objects):
            if j not in assigned_objs:
                next_active.append(TrackedEvent(str(uuid.uuid4()), time_step, time_step, [obj]))
                
        self.active_tracks = next_active
        
    def get_all_tracks(self) -> List[TrackedEvent]:
        return self.finished_tracks + self.active_tracks
