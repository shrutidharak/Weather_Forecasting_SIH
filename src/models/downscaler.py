"""
downscaler.py — Conditional Diffusion-Inspired Downscaler
==========================================================
Replaces the bilinear-interpolation baseline with a small conditional
denoising / super-resolution model.

Architecture: UNet-lite (3 encoder/decoder levels, skip connections)
Conditioned on: terrain layer (elevation + coastline_distance)

Training loss (logged separately so "extreme preservation" is demonstrable):
    total = MSE + λ_extreme × extreme_loss + λ_peak × peak_loss

    MSE:          standard pixel-level reconstruction
    extreme_loss: MSE restricted to cells above P90 of target (extra weight on tails)
    peak_loss:    max(0, true_max − pred_max)²  (penalise if sharpness is lost)

NOTE: "Diffusion-style" refers to the loss design (motivated by diffusion-model
score-matching, which also emphasises tails and sharp features) rather than a
full DDPM scheduler — that would require an order of magnitude more compute.
A DDPM/EDM scheduler can be slotted in here by wrapping ConvDownscaler as the
denoising network.

Real-data swap:
    Replace `generate_training_pairs()` with a dataloader over paired
    NCMRWF 12km→1km high-resolution tiles.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


# ── Bilinear baseline ──────────────────────────────────────────────────────────

class BilinearDownscaler:
    """
    Thin wrapper around scipy bilinear upsampling.
    Kept as the baseline comparator for the dashboard.
    """
    def __init__(self, scale_factor: int = 2):
        self.scale_factor = scale_factor

    def upscale(self, field: np.ndarray) -> np.ndarray:
        """
        field: (H, W) single-channel 2-D field.
        Returns (H*scale, W*scale).
        """
        from scipy.ndimage import zoom
        return zoom(field, self.scale_factor, order=1)  # bilinear


# ── UNet-lite building blocks ──────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class ConvDownscaler(nn.Module):
    """
    UNet-lite super-resolution model.
    Input:  (B, C_in + C_terrain, H, W)   — low-res field + terrain
    Output: (B, C_out, H*scale, W*scale)  — high-res prediction

    Default: C_in=1 (rainfall), C_terrain=2, C_out=1, scale=2
    """

    def __init__(self,
                 in_channels: int = 3,     # rainfall + 2 terrain
                 out_channels: int = 1,
                 base_ch: int = 16,
                 scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch,    base_ch * 2)
        self.enc3 = ConvBlock(base_ch*2,  base_ch * 4)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)

        # Decoder with skip connections
        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3  = ConvBlock(base_ch*8, base_ch*4)
        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2  = ConvBlock(base_ch*4, base_ch*2)
        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch,   2, stride=2)
        self.dec1  = ConvBlock(base_ch*2, base_ch)

        # Final upscale to target resolution
        if scale_factor == 2:
            self.final_up = nn.ConvTranspose2d(base_ch, out_channels, 2, stride=2)
        else:
            self.final_up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(base_ch, out_channels, 1),
            )
        self.final_act = nn.ReLU()   # rainfall is non-negative

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))

        # Bottleneck
        b  = self.bottleneck(F.avg_pool2d(e3, 2))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final_act(self.final_up(d1))


# ── Training loss components ───────────────────────────────────────────────────

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def extreme_value_loss(pred: torch.Tensor, target: torch.Tensor,
                       pct: float = 90.0) -> torch.Tensor:
    """
    Extra MSE weight on cells exceeding the P90 of the target.
    Ensures the model preserves tail behaviour, not just the mean.
    """
    threshold = torch.quantile(target.reshape(-1), pct / 100.0)
    mask = (target > threshold).float()
    sq_err = (pred - target) ** 2
    return (sq_err * mask).sum() / (mask.sum() + 1e-6)


def peak_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalise if the predicted maximum is below the true maximum.
    Prevents spatial blurring that smooth interpolation causes.
    """
    true_max = target.amax(dim=(-2, -1))   # (B,)
    pred_max = pred.amax(dim=(-2, -1))
    diff = F.relu(true_max - pred_max)     # only penalise under-prediction
    return (diff ** 2).mean()


def downscaler_loss(pred: torch.Tensor, target: torch.Tensor,
                    lambda_extreme: float = 2.0,
                    lambda_peak: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Combined loss with all three components logged separately.
    """
    l_mse     = mse_loss(pred, target)
    l_extreme = extreme_value_loss(pred, target)
    l_peak    = peak_loss(pred, target)
    total     = l_mse + lambda_extreme * l_extreme + lambda_peak * l_peak
    return {
        "total":   total,
        "mse":     l_mse,
        "extreme": l_extreme,
        "peak":    l_peak,
    }


# ── Synthetic training data generator ─────────────────────────────────────────

def generate_training_pairs(
    n_samples: int = 200,
    grid_h: int = 32, grid_w: int = 32,
    scale: int = 2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate (low_res, terrain, high_res) training pairs from synthetic data.

    Returns:
        lr:      (N, 1, H,   W)   — low-res rainfall
        terrain: (N, 2, H,   W)   — elevation + coast distance
        hr:      (N, 1, H*s, W*s) — high-res target

    Real-data swap: replace with a DataLoader over NetCDF tile pairs.
    """
    rng = np.random.default_rng(seed)

    # Import here to avoid circular dependency
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.data.synthetic_generator import SyntheticWeatherGenerator
    gen = SyntheticWeatherGenerator(grid_size=(grid_h, grid_w),
                                    time_steps=1, ensemble_members=1,
                                    downscale_factor=scale, seed=seed)

    lr_list, ter_list, hr_list = [], [], []
    terrain = gen.generate_terrain()   # (H, W, 2)
    terrain_t = torch.from_numpy(terrain.transpose(2, 0, 1).astype(np.float32))  # (2,H,W)
    # Normalise terrain
    terrain_t[0] /= 3000.0
    terrain_t[1] /= float(grid_w)

    for _ in range(n_samples):
        lr_s, hr_s, _ = gen.generate_scenario_a()
        # Take rainfall channel, ensemble 0, time 0
        lr_rain = lr_s[0, 0, :, :, 0].astype(np.float32)  # (H, W)
        hr_rain = hr_s[0, 0, :, :, 0].astype(np.float32)  # (H*s, W*s)

        lr_list.append(torch.from_numpy(lr_rain[np.newaxis]))   # (1,H,W)
        ter_list.append(terrain_t)
        hr_list.append(torch.from_numpy(hr_rain[np.newaxis]))   # (1,H*s,W*s)

    return (
        torch.stack(lr_list),   # (N,1,H,W)
        torch.stack(ter_list),  # (N,2,H,W)
        torch.stack(hr_list),   # (N,1,H*s,W*s)
    )


# ── Quick training loop ────────────────────────────────────────────────────────

def train_downscaler(
    model:          ConvDownscaler,
    n_samples:      int = 200,
    epochs:         int = 20,
    lr:             float = 1e-3,
    lambda_extreme: float = 2.0,
    lambda_peak:    float = 1.0,
    verbose:        bool  = True,
) -> Dict[str, list]:
    """
    Train ConvDownscaler on synthetic coarse→fine pairs.
    Logs mse, extreme, and peak losses separately for every epoch.

    Returns: history dict with lists per loss component.
    """
    lr_data, terrain_data, hr_data = generate_training_pairs(n_samples=n_samples)

    # Input = [low_res, terrain_ch0, terrain_ch1]
    inp = torch.cat([lr_data, terrain_data], dim=1)   # (N, 3, H, W)

    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"total": [], "mse": [], "extreme": [], "peak": []}

    model.train()
    for epoch in range(epochs):
        perm    = torch.randperm(len(inp))
        ep_loss = {k: 0.0 for k in history}

        # Mini-batch
        batch_size = 32
        for start in range(0, len(inp), batch_size):
            idx  = perm[start:start + batch_size]
            x    = inp[idx]
            y    = hr_data[idx]

            opt.zero_grad()
            pred   = model(x)
            losses = downscaler_loss(pred, y, lambda_extreme, lambda_peak)
            losses["total"].backward()
            opt.step()

            for k in ep_loss:
                ep_loss[k] += losses[k].item() * len(idx)

        for k in history:
            history[k].append(ep_loss[k] / len(inp))

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  [Downscaler] Epoch {epoch+1:3d}/{epochs}  "
                  f"total={history['total'][-1]:.4f}  "
                  f"mse={history['mse'][-1]:.4f}  "
                  f"extreme={history['extreme'][-1]:.4f}  "
                  f"peak={history['peak'][-1]:.4f}")

    model.eval()
    return history


class DownscalerFactory:
    """Returns a trained ConvDownscaler or BilinearDownscaler by name."""

    @staticmethod
    def build(name: str = "diffusion",
              train: bool = True, **kwargs) -> object:
        if name == "bilinear":
            return BilinearDownscaler()
        model = ConvDownscaler(**{k: v for k, v in kwargs.items()
                                  if k in ("in_channels", "out_channels",
                                           "base_ch", "scale_factor")})
        if train:
            print("Training ConvDownscaler on synthetic pairs...")
            train_downscaler(model,
                             epochs=kwargs.get("epochs", 20),
                             verbose=kwargs.get("verbose", True))
        return model


if __name__ == "__main__":
    print("=== Downscaler self-test ===")
    model = ConvDownscaler()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_downscaler(model, n_samples=50, epochs=10)
    print(f"Final total loss: {history['total'][-1]:.4f}")

    # Quick inference
    model.eval()
    with torch.no_grad():
        dummy = torch.rand(1, 3, 32, 32)
        out = model(dummy)
        print(f"Output shape: {out.shape}")   # expect (1, 1, 64, 64)
