"""
graph_builder.py — Spatio-Temporal Graph Construction
======================================================
Builds the graph structure fed into SpatioTemporalGNN.

Spatial edges:  4-connected lat/lon grid adjacency (N/S/E/W neighbours)
Temporal edges: same grid cell connected across consecutive time steps

NOTE: The lat/lon 4-connected grid here is a stand-in for a proper
icosahedral mesh (e.g., GraphWeather / Keisler 2022 mesh). To swap:
  1. Replace `build_spatial_edges()` with your mesh edge_index.
  2. Replace `node_positions` with icosahedral node lat/lon coordinates.
  3. All downstream code (SpatioTemporalGNN.forward) requires zero changes.

Real-data swap point:
  Use `torch_geometric` HeteroData or a custom edge_index built from the
  ECMWF 40962-node O96 octahedral reduced Gaussian grid.
"""
import numpy as np
from typing import Tuple, List


def build_spatial_edges(grid_h: int, grid_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    4-connected spatial adjacency on a (grid_h × grid_w) regular grid.

    Node index: i * grid_w + j  for grid position (i, j).

    Returns:
        edge_src: (E,) source node indices
        edge_dst: (E,) destination node indices
    """
    srcs, dsts = [], []
    for i in range(grid_h):
        for j in range(grid_w):
            node = i * grid_w + j
            # North
            if i > 0:
                srcs.append(node); dsts.append((i - 1) * grid_w + j)
            # South
            if i < grid_h - 1:
                srcs.append(node); dsts.append((i + 1) * grid_w + j)
            # West
            if j > 0:
                srcs.append(node); dsts.append(i * grid_w + j - 1)
            # East
            if j < grid_w - 1:
                srcs.append(node); dsts.append(i * grid_w + j + 1)
    return np.array(srcs, dtype=np.int64), np.array(dsts, dtype=np.int64)


def build_temporal_edges(n_nodes: int, n_timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Temporal edges connecting each node at time t to the same node at t+1.

    The combined node index in the spatio-temporal graph is:
        t * n_nodes + node_spatial_idx

    Returns:
        edge_src: (T-1) * n_nodes source indices
        edge_dst: (T-1) * n_nodes destination indices
    """
    srcs, dsts = [], []
    for t in range(n_timesteps - 1):
        for n in range(n_nodes):
            srcs.append(t * n_nodes + n)
            dsts.append((t + 1) * n_nodes + n)
    return np.array(srcs, dtype=np.int64), np.array(dsts, dtype=np.int64)


def build_spatiotemporal_graph(
    grid_h: int,
    grid_w: int,
    n_timesteps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full spatio-temporal graph combining spatial and temporal edges.

    Returns:
        sp_src, sp_dst: spatial edge arrays (referencing node indices within one timestep)
        tp_src, tp_dst: temporal edge arrays (referencing flattened T*H*W node space)
    """
    sp_src, sp_dst = build_spatial_edges(grid_h, grid_w)
    tp_src, tp_dst = build_temporal_edges(grid_h * grid_w, n_timesteps)
    return sp_src, sp_dst, tp_src, tp_dst


def node_positions(grid_h: int, grid_w: int) -> np.ndarray:
    """
    Returns (n_nodes, 2) array of (lat_idx, lon_idx) for each node.
    Used as positional features in node embeddings.
    """
    idxs = []
    for i in range(grid_h):
        for j in range(grid_w):
            idxs.append([i / grid_h, j / grid_w])   # normalised [0,1]
    return np.array(idxs, dtype=np.float32)


if __name__ == "__main__":
    H, W, T = 4, 4, 3
    sp_s, sp_d = build_spatial_edges(H, W)
    tp_s, tp_d = build_temporal_edges(H * W, T)
    print(f"Grid {H}x{W}, T={T}")
    print(f"Spatial edges: {len(sp_s)} (expect ~2*H*W*4/2 undirected → directed)")
    print(f"Temporal edges: {len(tp_s)} (expect (T-1)*H*W = {(T-1)*H*W})")
    pos = node_positions(H, W)
    print(f"Node positions shape: {pos.shape}")
