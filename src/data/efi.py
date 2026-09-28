"""
efi.py — Extreme Forecast Index (EFI) Scorer
=============================================
Computes a Mann-Whitney-style EFI in [−1, +1] per grid cell,
per variable, given:
  • A forecast ensemble distribution
  • The stored synthetic climatological distribution (from SyntheticClimatology)

Formula (ECMWF-inspired closed-form approximation):
    EFI(i,j) = (2/π) × arcsin(
        (F_fc − F_clim) /
        sqrt(F_fc(1−F_fc) + F_clim(1−F_clim) + ε)
    )

where:
    F_fc   = fraction of ensemble members exceeding the climatological median
             at grid cell (i,j)
    F_clim = 0.5 (climatological median by definition)

Interpretation:
    EFI ~ +1  → almost all ensemble members exceed what climatology ever saw
    EFI ~  0  → forecast matches climatology
    EFI ~ -1  → almost all members are below the climatological minimum

Real-data swap:
    Replace `climatology.percentile_*` arrays with per-cell empirical CDFs
    derived from NCMRWF NEPS-G hindcast archive.
"""
import numpy as np
from typing import Optional

try:
    from .climatology import SyntheticClimatology
except ImportError:
    from climatology import SyntheticClimatology


class EFIScorer:
    """
    Vectorised EFI computation over a full (lat, lon) grid.

    Usage:
        scorer = EFIScorer(climatology)
        # ensemble_forecast: (n_members, lat, lon)
        efi_grid = scorer.compute(ensemble_forecast, feature_idx=0)
        # efi_grid: (lat, lon)  values in [-1, 1]
        anomaly_mask = efi_grid > threshold  # e.g. threshold=0.5
    """

    def __init__(self, climatology: SyntheticClimatology,
                 threshold: float = 0.5):
        """
        Args:
            climatology: A fitted SyntheticClimatology instance.
            threshold:   EFI value above which a grid cell triggers detection.
                         0.5 is a commonly used operational value.
        """
        self.clim = climatology
        self.threshold = threshold
        self._eps = 1e-6

    def compute(self, ensemble_forecast: np.ndarray,
                feature_idx: int = 0) -> np.ndarray:
        """
        Compute EFI for a single variable over the full grid.

        Args:
            ensemble_forecast: (n_members, lat, lon) — raw forecast values
                               for ONE variable (not standardised).
            feature_idx:       Which climatology feature to compare against.

        Returns:
            efi: (lat, lon) array in [-1, 1].
        """
        if self.clim.mean is None:
            raise ValueError("Climatology not fitted — call generate_historical_baseline().")

        n_members, H, W = ensemble_forecast.shape

        # Climatological median as the reference threshold
        # Use the interpolated 50th percentile (mean of p25 and p75 knots)
        clim_median = 0.5 * (
            self.clim.percentile_25[..., feature_idx] +
            self.clim.percentile_75[..., feature_idx]
        )  # (H, W)

        # F_fc: fraction of ensemble members exceeding climatological median
        exceed = (ensemble_forecast > clim_median[np.newaxis]).sum(axis=0)  # (H, W)
        F_fc = np.clip(exceed / n_members, self._eps, 1 - self._eps)        # (H, W)

        # F_clim = 0.5 (median ↔ 50th percentile)
        F_clim = 0.5

        numerator   = F_fc - F_clim
        denominator = np.sqrt(
            F_fc   * (1 - F_fc) +
            F_clim * (1 - F_clim) +
            self._eps
        )

        efi = (2.0 / np.pi) * np.arcsin(numerator / denominator)
        return np.clip(efi, -1.0, 1.0)

    def get_anomaly_mask(self, ensemble_forecast: np.ndarray,
                         feature_idx: int = 0) -> np.ndarray:
        """
        Returns boolean mask where EFI exceeds self.threshold.
        Drop-in replacement for the old percentile-threshold exceedance mask,
        with spatial context built into the score.
        """
        efi = self.compute(ensemble_forecast, feature_idx)
        return efi > self.threshold

    def compute_all_features(self, ensemble_forecast: np.ndarray) -> np.ndarray:
        """
        Compute EFI for every feature channel.

        Args:
            ensemble_forecast: (n_members, lat, lon, n_features)

        Returns:
            efi_all: (lat, lon, n_features) — EFI per channel.
        """
        n_feat = ensemble_forecast.shape[-1]
        results = []
        for f in range(n_feat):
            efi = self.compute(ensemble_forecast[..., f], feature_idx=f)
            results.append(efi)
        return np.stack(results, axis=-1)

    def probability_of_exceedance(self, ensemble_forecast: np.ndarray,
                                  threshold_value: float,
                                  feature_idx: int = 0) -> np.ndarray:
        """
        Fraction of ensemble members exceeding an absolute threshold.
        Used to populate `probability_exceedance` in the TrackedEvent schema.

        Args:
            ensemble_forecast: (n_members, lat, lon)
            threshold_value:   Absolute value to exceed (e.g. 50 mm/hr)
            feature_idx:       Unused here but kept for interface consistency.

        Returns:
            prob: (lat, lon) in [0, 1]
        """
        exceed = (ensemble_forecast > threshold_value).sum(axis=0)
        return exceed / ensemble_forecast.shape[0]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    from data.climatology import SyntheticClimatology
    from data.synthetic_generator import SyntheticDataLoader

    clim = SyntheticClimatology()
    clim.generate_historical_baseline()

    loader = SyntheticDataLoader(scenario="cyclone", seed=0)
    lr, _, _ = loader.load_data()

    scorer = EFIScorer(clim, threshold=0.5)
    # Use all 10 ensemble members, time=5, rainfall channel
    fc_ensemble = lr[:, 5, :, :, 0]   # (10, 32, 32)
    efi = scorer.compute(fc_ensemble, feature_idx=0)
    print(f"EFI shape: {efi.shape}, range: [{efi.min():.3f}, {efi.max():.3f}]")
    mask = scorer.get_anomaly_mask(fc_ensemble)
    print(f"Anomaly cells (EFI > 0.5): {mask.sum()}")
