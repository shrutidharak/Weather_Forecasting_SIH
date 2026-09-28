import numpy as np
from typing import Tuple, Dict

class SyntheticClimatology:
    def __init__(self, 
                 grid_size: Tuple[int, int] = (32, 32),
                 features: int = 6,             # updated: 6 channels
                 historical_years: int = 10):   # 10 years of simulated background
        self.grid_size = grid_size
        self.features = features
        self.historical_years = historical_years
        
        # Will store the climatological arrays
        self.mean: np.ndarray = None
        self.std: np.ndarray = None
        # Original percentile arrays (kept for BaselineDetector backward-compat)
        self.percentile_90: np.ndarray = None
        self.percentile_95: np.ndarray = None
        self.percentile_99: np.ndarray = None
        # Extended percentile arrays used by EFIScorer
        self.percentile_10: np.ndarray = None
        self.percentile_25: np.ndarray = None
        self.percentile_75: np.ndarray = None

    def generate_historical_baseline(self):
        """
        Generates a purely synthetic historical dataset containing mostly noise,
        with occasional randomized extreme values to simulate natural variation.
        This prevents data leakage by generating entirely separate baseline data.
        """
        # Let's say 365 days * 10 years = 3650 time steps
        time_steps = 365 * self.historical_years
        
        # Base noise matches the generator's background noise scale
        historical_data = np.random.normal(loc=0.0, scale=0.1, size=(time_steps, self.grid_size[0], self.grid_size[1], self.features))
        
        # Introduce some random "weather events" to give the climatology realistic fat tails
        # We'll just randomly spike some pixels
        num_spikes = time_steps * 5
        for _ in range(num_spikes):
            t = np.random.randint(0, time_steps)
            lat = np.random.randint(0, self.grid_size[0])
            lon = np.random.randint(0, self.grid_size[1])
            f = np.random.randint(0, self.features)
            historical_data[t, lat, lon, f] += np.random.exponential(scale=5.0)

        # Calculate statistics across the time dimension (axis 0)
        self.mean = np.mean(historical_data, axis=0)
        self.std = np.std(historical_data, axis=0)
        
        self.percentile_10 = np.percentile(historical_data, 10, axis=0)
        self.percentile_25 = np.percentile(historical_data, 25, axis=0)
        self.percentile_75 = np.percentile(historical_data, 75, axis=0)
        self.percentile_90 = np.percentile(historical_data, 90, axis=0)
        self.percentile_95 = np.percentile(historical_data, 95, axis=0)
        self.percentile_99 = np.percentile(historical_data, 99, axis=0)

    def get_clim_cdf(self, value: np.ndarray, feature_idx: int) -> np.ndarray:
        """
        Approximate climatological CDF for EFI computation.
        Uses a piecewise-linear interpolation over stored percentile knots.
        Returns CDF values in [0, 1] matching the shape of `value`.
        """
        if self.mean is None:
            raise ValueError("Call generate_historical_baseline() first.")
        pcts = np.array([0.10, 0.25, 0.75, 0.90, 0.95, 0.99])
        knots = np.stack([
            self.percentile_10[..., feature_idx],
            self.percentile_25[..., feature_idx],
            self.percentile_75[..., feature_idx],
            self.percentile_90[..., feature_idx],
            self.percentile_95[..., feature_idx],
            self.percentile_99[..., feature_idx],
        ], axis=0)   # (6, lat, lon)
        # Interpolate per cell using broadcasting
        cdf = np.interp(value.ravel(),
                        knots[:, value.shape[0]//2, value.shape[1]//2],  # grid-mean knots
                        pcts)
        return np.clip(cdf.reshape(value.shape), 1e-6, 1 - 1e-6)

    def get_standardized_anomaly(self, forecast: np.ndarray) -> np.ndarray:
        """
        Converts a raw forecast into standardized anomalies (z-scores).
        forecast shape: (ensembles, time, lat, lon, features) or (time, lat, lon, features)
        """
        if self.mean is None or self.std is None:
            raise ValueError("Climatology not generated yet. Call generate_historical_baseline().")
        
        # Handle division by zero
        safe_std = np.where(self.std == 0, 1e-6, self.std)
        
        # Broadcasting should handle (ensembles, time, lat, lon, feat) - (lat, lon, feat)
        return (forecast - self.mean) / safe_std
        
    def get_exceedance_mask(self, forecast: np.ndarray, percentile: int = 95) -> np.ndarray:
        """
        Returns a boolean mask where the forecast exceeds the historical percentile threshold.
        """
        if percentile == 90:
            threshold = self.percentile_90
        elif percentile == 95:
            threshold = self.percentile_95
        elif percentile == 99:
            threshold = self.percentile_99
        else:
            raise NotImplementedError("Only 90, 95, 99 percentiles are cached.")
            
        return forecast > threshold

if __name__ == "__main__":
    climatology = SyntheticClimatology()
    print("Generating synthetic climatology...")
    climatology.generate_historical_baseline()
    print(f"Mean shape: {climatology.mean.shape}")
    print(f"99th Percentile max value (Feature 0): {climatology.percentile_99[:, :, 0].max():.2f}")
