"""
physics_loss.py — Physics-Informed Constraint Module
======================================================
Implements one concrete, checkable physical consistency constraint:

    RULE: Predicted rainfall should not be high while the co-located
          moisture-convergence proxy (derived from wind + humidity) is low.

    moisture_convergence = –humidity × divergence(u, v)
    where divergence = ∂u/∂x + ∂v/∂y

    High rainfall + low convergence = physically implausible.

Exposes two interfaces:
1. `moisture_convergence_constraint()`  — additive loss term for downscaler training
2. `physics_plausibility_score()`       — inference-time [0,1] scalar for dashboard/API

Real-data swap:
    Replace finite-difference divergence with proper spherical divergence
    on the native model grid (e.g., using metpy or xarray-based operators
    on pressure-level NCMRWF NEPS-G output).
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


# ── NumPy version (used at inference / in metrics) ────────────────────────────

def _divergence_np(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Finite-difference divergence: ∂u/∂x + ∂v/∂y
    u, v: (..., H, W) arrays
    Returns (..., H, W)
    """
    du_dx = np.gradient(u, axis=-1)
    dv_dy = np.gradient(v, axis=-2)
    return du_dx + dv_dy


def _moisture_convergence_np(u: np.ndarray, v: np.ndarray,
                              humidity: np.ndarray) -> np.ndarray:
    """
    Moisture convergence proxy: −humidity × divergence(u, v)
    Positive = convergence (moisture influx — rainfall likely).
    """
    return -humidity * _divergence_np(u, v)


def physics_plausibility_score(
    rainfall:  np.ndarray,   # (..., H, W)
    wind_u:    np.ndarray,   # (..., H, W)
    wind_v:    np.ndarray,   # (..., H, W)
    humidity:  np.ndarray,   # (..., H, W)
    rain_threshold_pct: float = 75.0,
    conv_threshold: float = 0.0,
) -> float:
    """
    Compute a physics plausibility score in [0, 1].

    Score = fraction of heavy-rain cells that ALSO have positive
            moisture convergence (physically consistent).

    A score of 1.0 means every heavy-rain cell is supported by
    moisture convergence. A score near 0 means the forecast is
    physically implausible (rainfall with no moisture supply).

    Args:
        rainfall:  predicted or observed rainfall field
        wind_u/v:  u/v wind components (same grid)
        humidity:  specific/relative humidity (same grid)
        rain_threshold_pct: percentile above which cells are "heavy rain"
        conv_threshold: minimum convergence to be considered "supported"

    Returns:
        scalar float in [0, 1]
    """
    conv = _moisture_convergence_np(wind_u, wind_v, humidity)

    # Binary mask of heavy-rain cells
    rain_thresh = np.percentile(rainfall, rain_threshold_pct)
    heavy_rain  = rainfall > rain_thresh

    if heavy_rain.sum() == 0:
        return 1.0   # no heavy rain → trivially plausible

    # Fraction of heavy-rain cells with positive convergence
    plausible = (conv[heavy_rain] > conv_threshold).sum()
    score = float(plausible) / float(heavy_rain.sum())
    return float(np.clip(score, 0.0, 1.0))


# ── PyTorch version (used during downscaler training) ─────────────────────────

def _divergence_torch(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Finite-difference divergence for 4-D tensors (B, 1, H, W).
    Uses central differences via conv2d.
    """
    # Kernels: ∂/∂x (longitude) and ∂/∂y (latitude)
    kx = torch.tensor([[[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]]],
                      dtype=u.dtype, device=u.device)
    ky = torch.tensor([[[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]]],
                      dtype=u.dtype, device=u.device)
    du_dx = F.conv2d(u, kx, padding=1)
    dv_dy = F.conv2d(v, ky, padding=1)
    return du_dx + dv_dy


def moisture_convergence_constraint(
    rainfall_pred: torch.Tensor,   # (B, 1, H, W) — predicted rainfall
    wind_u:        torch.Tensor,   # (B, 1, H, W)
    wind_v:        torch.Tensor,   # (B, 1, H, W)
    humidity:      torch.Tensor,   # (B, 1, H, W)
    rain_thresh:   float = 5.0,    # mm/hr threshold for "heavy rain"
    weight:        float = 1.0,    # loss weighting coefficient
) -> torch.Tensor:
    """
    Physics-informed loss term: penalise predicted rainfall that is high
    while moisture convergence (from co-located wind + humidity) is low.

    Loss = mean over (heavy-rain cells) of max(0, –convergence)²
         = encourages convergence to be positive wherever it rains heavily.

    This term is ADDITIVE to the downscaler's reconstruction loss.

    Args:
        rainfall_pred: predicted high-res rainfall
        wind_u/v:      wind components at SAME resolution (bilinearly upsampled
                       from low-res if needed — caller's responsibility)
        humidity:      humidity at same resolution
        rain_thresh:   absolute rainfall threshold for "heavy rain" mask
        weight:        scalar multiplier on the returned loss

    Returns:
        Scalar loss tensor.
    """
    conv = -humidity * _divergence_torch(wind_u, wind_v)  # (B, 1, H, W)

    # Heavy-rain mask (soft, differentiable via sigmoid)
    heavy = torch.sigmoid((rainfall_pred - rain_thresh) * 2.0)   # ~1 where heavy

    # Penalise: where it rains heavily AND convergence is NEGATIVE (wrong sign)
    violation = torch.relu(-conv) * heavy
    loss = (violation ** 2).mean()
    return weight * loss


# ── Combined wrapper for training ─────────────────────────────────────────────

def physics_loss_for_training(
    rainfall_pred: torch.Tensor,
    wind_u:        torch.Tensor,
    wind_v:        torch.Tensor,
    humidity:      torch.Tensor,
    weight:        float = 0.5,
) -> torch.Tensor:
    """
    Convenience wrapper used in downscaler training.
    Returns a scalar loss tensor ready for .backward().
    """
    return moisture_convergence_constraint(
        rainfall_pred, wind_u, wind_v, humidity, weight=weight
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.data.synthetic_generator import SyntheticDataLoader, SyntheticWeatherGenerator

    loader = SyntheticDataLoader(scenario="cyclone", seed=0)
    lr, _, _ = loader.load_data()

    # Test on ensemble mean, time 10
    frame = lr[:, 10].mean(axis=0)   # (H, W, 6)
    rain  = frame[:, :, 0]
    u     = frame[:, :, 1]
    v     = frame[:, :, 2]
    hum   = frame[:, :, 5]

    score = physics_plausibility_score(rain, u, v, hum)
    print(f"Physics plausibility score (cyclone T=10): {score:.3f}")

    # PyTorch version
    rain_t = torch.from_numpy(rain[np.newaxis, np.newaxis].astype("float32"))
    u_t    = torch.from_numpy(u[np.newaxis, np.newaxis].astype("float32"))
    v_t    = torch.from_numpy(v[np.newaxis, np.newaxis].astype("float32"))
    hum_t  = torch.from_numpy(hum[np.newaxis, np.newaxis].astype("float32"))

    loss = moisture_convergence_constraint(rain_t, u_t, v_t, hum_t)
    print(f"Physics constraint loss (PyTorch): {loss.item():.4f}")
