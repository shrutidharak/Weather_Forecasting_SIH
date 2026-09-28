"""
st_gnn.py — Spatio-Temporal GNN (Main Module)
=============================================
THE CENTREPIECE of the SIH weather prototype.

Processes a multi-variable weather grid across a time window using a
spatio-temporal graph neural network. Anomaly detection, event typing,
and 4D bounding box prediction all happen INSIDE this module.

Architecture:
    Input  : (B, T, H, W, C)  — batch, time, height, width, channels
    Graph  : each (t, i, j) cell is a node
             spatial edges  : 4-connected grid per timestep
             temporal edges : same cell, t → t+1
    GNN    : 3 rounds of message passing
             round 1 — spatial neighbour aggregation
             round 2 — temporal (prev/next) aggregation
             round 3 — combined spatial+temporal update
    Heads  : EventTypeHead  → softmax over 5 event types
             4DBoundingBoxHead → bbox + severity + confidence

NOTE: The spatial graph is a regular lat/lon 4-connected grid — a stand-in
for an icosahedral mesh (e.g., Keisler 2022 / GraphWeather). To swap,
replace `graph_builder.build_spatial_edges()` with your mesh edge_index.
`SpatioTemporalGNN.forward()` requires zero changes.

Real-data swap:
    Replace SyntheticDataLoader input with a RealWeatherDataLoader that
    parses xarray/NetCDF NCMRWF NEPS-G tensors of the same (B,T,H,W,C) shape.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    from .graph_builder import build_spatial_edges, node_positions
    from .bbox_head import BoundingBoxHead, Event4DBoundingBox, EVENT_TYPE_NAMES
except ImportError:
    from graph_builder import build_spatial_edges, node_positions
    from bbox_head import BoundingBoxHead, Event4DBoundingBox, EVENT_TYPE_NAMES


# ── Node feature dimension breakdown ─────────────────────────────────────────
# Raw channels:       6   (rainfall, u, v, temp, pressure, humidity)
# Positional (lat/lon):2  (normalised grid coords)
# Terrain:            2   (elevation, coastline_dist — normalised)
# EFI score:          1   (scalar per cell from EFIScorer)
# ─────────────────────────────────────────
# Total node_dim:    11
NODE_DIM   = 11
HIDDEN_DIM = 32


class SpatialMessagePass(nn.Module):
    """
    One round of spatial message passing.
    Each node aggregates mean of its 4-connected spatial neighbours,
    then updates its embedding via an MLP.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor,
                sp_src: torch.Tensor, sp_dst: torch.Tensor) -> torch.Tensor:
        """
        x     : (N, in_dim)  — node features (all nodes in one timestep)
        sp_src: (E,)         — source indices
        sp_dst: (E,)         — destination indices
        Returns (N, out_dim)
        """
        N = x.size(0)
        # Aggregate neighbour messages (mean pooling)
        agg = torch.zeros(N, x.size(1), device=x.device)
        count = torch.zeros(N, 1, device=x.device)
        msg = x[sp_src]          # (E, D)
        agg.index_add_(0, sp_dst, msg)
        count.index_add_(0, sp_dst, torch.ones(len(sp_src), 1, device=x.device))
        count = count.clamp(min=1)
        agg = agg / count        # mean
        combined = torch.cat([x, agg], dim=-1)   # (N, 2D)
        return self.update(combined)


class TemporalMessagePass(nn.Module):
    """
    One round of temporal message passing.
    Each node at time t aggregates from the same node at t-1 and t+1.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, T: int, N_per_t: int) -> torch.Tensor:
        """
        x: (T*N, in_dim)  — all nodes across all timesteps
        Returns (T*N, out_dim)
        """
        total = x.size(0)
        agg   = torch.zeros_like(x)
        count = torch.zeros(total, 1, device=x.device)

        for t in range(T):
            start = t * N_per_t
            end   = (t + 1) * N_per_t
            # Aggregate from t-1
            if t > 0:
                agg[start:end]   += x[(t-1)*N_per_t:t*N_per_t]
                count[start:end] += 1
            # Aggregate from t+1
            if t < T - 1:
                agg[start:end]   += x[(t+1)*N_per_t:(t+2)*N_per_t]
                count[start:end] += 1
        count = count.clamp(min=1)
        agg = agg / count
        combined = torch.cat([x, agg], dim=-1)
        return self.update(combined)


class SpatioTemporalGNN(nn.Module):
    """
    Main Spatio-Temporal GNN module.

    Three rounds of message passing:
      Round 1 (spatial):   incorporates spatial neighbourhood context
      Round 2 (temporal):  incorporates temporal evolution context
      Round 3 (combined):  final spatial pass on temporally-aware embeddings

    Outputs per grid cell:
      • node_embeddings:  (B, T, N, hidden_dim)
      • event_type_logits:(B, T, N, 5)
      • detection_scores: (B, T, N)  — in [0,1], used as objectness for bbox head
    """
    def __init__(self,
                 node_dim:   int = NODE_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Round 1 — spatial
        self.sp_pass_1 = SpatialMessagePass(hidden_dim, hidden_dim)
        # Round 2 — temporal
        self.tp_pass   = TemporalMessagePass(hidden_dim, hidden_dim)
        # Round 3 — spatial again (on temporally-aware features)
        self.sp_pass_2 = SpatialMessagePass(hidden_dim, hidden_dim)

        # Residual projection (for skip connections)
        self.skip = nn.Linear(hidden_dim, hidden_dim)

        # Event type classification head
        self.event_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(EVENT_TYPE_NAMES)),
        )

        # Detection score head (objectness)
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_features: torch.Tensor,  # (B, T, N, node_dim)
        sp_src: torch.Tensor,         # (E,)  spatial edge sources
        sp_dst: torch.Tensor,         # (E,)  spatial edge destinations
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: (B, T, N, node_dim)
            sp_src/sp_dst: precomputed spatial edge indices (same for each t)

        Returns dict with keys:
            embeddings:        (B, T, N, hidden_dim)
            event_type_logits: (B, T, N, 5)
            event_type_probs:  (B, T, N, 5)
            detection_scores:  (B, T, N)
        """
        B, T, N, _ = node_features.shape

        # Project all nodes at once
        x = self.input_proj(node_features.view(B * T * N, -1))    # (B*T*N, H)
        x = x.view(B, T * N, -1)   # (B, T*N, H)

        results = {}
        embs_list = []

        for b in range(B):
            xb = x[b]   # (T*N, H)

            # ── Round 1: Spatial (per-timestep) ──
            sp_out = []
            for t in range(T):
                xt = xb[t*N:(t+1)*N]    # (N, H)
                xt2 = self.sp_pass_1(xt, sp_src, sp_dst)
                sp_out.append(xt2)
            xb = torch.cat(sp_out, dim=0)   # (T*N, H)

            # ── Round 2: Temporal ──
            xb = self.tp_pass(xb, T, N)    # (T*N, H)

            # ── Round 3: Spatial again ──
            sp_out2 = []
            for t in range(T):
                xt = xb[t*N:(t+1)*N]
                xt2 = self.sp_pass_2(xt, sp_src, sp_dst) + self.skip(xt)  # residual
                sp_out2.append(xt2)
            xb = torch.cat(sp_out2, dim=0)   # (T*N, H)

            embs_list.append(xb.view(T, N, -1))

        embeddings = torch.stack(embs_list, dim=0)   # (B, T, N, H)

        # Heads
        flat = embeddings.view(B * T * N, -1)
        et_logits = self.event_type_head(flat).view(B, T, N, -1)
        et_probs  = F.softmax(et_logits, dim=-1)
        det_score = self.detection_head(flat).view(B, T, N)

        return {
            "embeddings":        embeddings,
            "event_type_logits": et_logits,
            "event_type_probs":  et_probs,
            "detection_scores":  det_score,
        }


# ── Full pipeline wrapper ──────────────────────────────────────────────────────

class SpatioTemporalEventDetector:
    """
    Wraps SpatioTemporalGNN + BoundingBoxHead into a single callable
    that takes raw numpy arrays and returns List[Event4DBoundingBox].

    Handles:
      • Feature assembly (raw channels + positional + terrain + EFI)
      • ST-GNN forward pass
      • BoundingBoxHead event extraction + Hungarian matching
      • Physics plausibility scoring (delegated to PhysicsLoss module)
    """

    def __init__(self,
                 grid_h: int, grid_w: int,
                 model: Optional[SpatioTemporalGNN] = None,
                 detection_threshold: float = 0.3,
                 efi_threshold: float = 0.5):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.detection_threshold = detection_threshold
        self.efi_threshold = efi_threshold

        self.model = model or SpatioTemporalGNN()
        self.model.eval()

        # Precompute spatial edge indices
        sp_src_np, sp_dst_np = build_spatial_edges(grid_h, grid_w)
        self.sp_src = torch.from_numpy(sp_src_np)
        self.sp_dst = torch.from_numpy(sp_dst_np)

        # Node positions (fixed for this grid)
        self._pos = torch.from_numpy(node_positions(grid_h, grid_w))  # (N, 2)

        self.bbox_head = BoundingBoxHead(grid_h, grid_w)

    def _assemble_node_features(
        self,
        data:    np.ndarray,     # (T, H, W, C) — one ensemble member
        terrain: np.ndarray,     # (H, W, 2)
        efi:     np.ndarray,     # (H, W) or (T, H, W)  EFI score for rain channel
    ) -> torch.Tensor:
        """
        Assemble (T, N, node_dim) tensor from raw data + terrain + EFI.
        node_dim = 6 (channels) + 2 (position) + 2 (terrain) + 1 (EFI) = 11
        """
        T, H, W, C = data.shape
        N = H * W

        # Normalise channels to roughly [-1, 1]
        norms = np.array([20.0, 30.0, 30.0, 40.0, 10.0, 1.0])   # per-channel scale
        data_n = data / (norms[np.newaxis, np.newaxis, np.newaxis, :] + 1e-6)

        # Terrain normalisation
        terrain_n = terrain.copy()
        terrain_n[..., 0] /= 3000.0   # elevation
        terrain_n[..., 1] /= max(W, 1)  # coast dist

        # EFI: broadcast to (T, H, W) if scalar grid
        if efi.ndim == 2:
            efi_t = np.broadcast_to(efi[np.newaxis], (T, H, W))
        else:
            efi_t = efi

        # Build (T, N, node_dim)
        parts = []
        for t in range(T):
            raw = data_n[t].reshape(N, C)                           # (N, 6)
            pos = self._pos.numpy()                                  # (N, 2)
            ter = terrain_n.reshape(N, 2)                           # (N, 2)
            efi_col = efi_t[t].reshape(N, 1)                       # (N, 1)
            node = np.concatenate([raw, pos, ter, efi_col], axis=1) # (N, 11)
            parts.append(node)

        arr = np.stack(parts, axis=0)   # (T, N, 11)
        return torch.from_numpy(arr.astype(np.float32))

    @torch.no_grad()
    def detect(
        self,
        data:       np.ndarray,   # (E, T, H, W, C) — full ensemble
        terrain:    np.ndarray,   # (H, W, 2)
        efi_grids:  np.ndarray,   # (E, H, W) — EFI for rainfall channel
        forecast_lead: int = 0,
        physics_plausibility: float = 1.0,
    ) -> List[Event4DBoundingBox]:
        """
        Run the full ST-GNN detection pipeline.

        Args:
            data:       (E, T, H, W, C) ensemble tensor
            terrain:    (H, W, 2) terrain layer
            efi_grids:  (E, H, W) precomputed EFI per ensemble member
            forecast_lead: lead time in hours (for schema output)
            physics_plausibility: physics score from PhysicsLoss (0–1)

        Returns:
            List of Event4DBoundingBox objects for the detected events.
        """
        E, T, H, W, C = data.shape

        # Build node features for ensemble mean (representative member)
        mean_data = data.mean(axis=0)   # (T, H, W, C)
        mean_efi  = efi_grids.mean(axis=0)   # (H, W)

        node_feat = self._assemble_node_features(mean_data, terrain, mean_efi)
        # Add batch dim: (1, T, N, D)
        node_feat_batch = node_feat.unsqueeze(0)

        out = self.model(node_feat_batch, self.sp_src, self.sp_dst)

        det_scores  = out["detection_scores"][0]    # (T, N)
        et_probs    = out["event_type_probs"][0]    # (T, N, 5)
        embeddings  = out["embeddings"][0]          # (T, N, H)

        # Compute ensemble confidence: fraction of members that show EFI > 0.5
        ensemble_confidence = float((efi_grids > self.efi_threshold).mean())

        events = self.bbox_head.extract_events(
            detection_scores=det_scores,
            event_type_probs=et_probs,
            detection_threshold=self.detection_threshold,
            terrain=terrain,
            raw_data=mean_data,
            forecast_lead=forecast_lead,
            ensemble_confidence=ensemble_confidence,
            physics_plausibility=physics_plausibility,
        )
        return events


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.data.synthetic_generator import SyntheticDataLoader
    from src.data.climatology import SyntheticClimatology
    from src.data.efi import EFIScorer

    loader   = SyntheticDataLoader(scenario="cyclone", seed=0)
    lr, _, _ = loader.load_data()
    terrain  = loader.get_terrain()

    clim = SyntheticClimatology()
    clim.generate_historical_baseline()
    scorer = EFIScorer(clim)

    efi_grids = np.stack([
        scorer.compute(lr[:, t, :, :, 0], feature_idx=0)
        for t in range(lr.shape[1])
    ], axis=0)   # (T, H, W) — use time 0 for demo

    detector = SpatioTemporalEventDetector(32, 32)
    events   = detector.detect(lr, terrain, efi_grids[0:1].repeat(10, axis=0))

    print(f"\nDetected {len(events)} events:")
    for ev in events[:3]:
        print(f"  {ev['event_id'][:8]}… type={ev['event_type']}  "
              f"lat={ev['latitude_range']}  sev={ev['severity']:.2f}  "
              f"risk={ev['risk_level']}")
