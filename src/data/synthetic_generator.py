"""
synthetic_generator.py
======================
Generates multi-variable synthetic weather grids for the SIH prototype.

Channels (feature index):
  0: rainfall          (mm/hr)
  1: wind_u            (m/s, eastward)
  2: wind_v            (m/s, northward)
  3: temperature       (°C anomaly)
  4: pressure          (hPa anomaly, negative = low pressure)
  5: humidity          (0–1 fractional)

Derived quantities (helpers, NOT extra channels):
  wind_speed(data)            = sqrt(u² + v²)
  moisture_convergence(data)  = –humidity × (∂u/∂x + ∂v/∂y)

Terrain layer (generated once per grid):
  terrain[:, :, 0] = elevation (m, 0–3000)
  terrain[:, :, 1] = coastline_distance (grid cells, 0–grid_max)

Real-data swap point:
  Replace SyntheticDataLoader.load_data() with RealWeatherDataLoader that
  parses xarray/NetCDF tensors from NCMRWF NEPS-G. Downstream pipeline
  (ST-GNN, tracker, downscaler) requires no changes.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import uuid

# ── Channel indices ────────────────────────────────────────────────────────────
CH_RAIN  = 0
CH_U     = 1
CH_V     = 2
CH_TEMP  = 3
CH_PRES  = 4
CH_HUM   = 5
N_CHANNELS = 6

# ── Event type labels ──────────────────────────────────────────────────────────
EVENT_TYPES = ["cyclone", "heatwave", "cold_wave", "extreme_rainfall", "normal"]


@dataclass
class EventGroundTruth:
    event_id: str
    event_type: str          # one of EVENT_TYPES
    timestamp: int
    centroid_lat: float
    centroid_lon: float
    intensity: float         # peak rainfall (mm/hr) in the event blob
    velocity_lat: float
    velocity_lon: float
    bbox: Tuple[float, float, float, float]   # min_lat, min_lon, max_lat, max_lon
    pressure_min_hpa: float = 1010.0          # synthetic vertical range
    pressure_max_hpa: float = 1013.0


class SyntheticWeatherGenerator:
    """
    Generates correlated multi-variable weather grids.

    Output tensor shape: (ensembles, time_steps, lat, lon, N_CHANNELS)
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (32, 32),
                 time_steps: int = 40,
                 ensemble_members: int = 10,
                 downscale_factor: int = 2,
                 seed: Optional[int] = None):
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.ensemble_members = ensemble_members
        self.downscale_factor = downscale_factor
        self.high_res_grid = (grid_size[0] * downscale_factor,
                              grid_size[1] * downscale_factor)
        self.rng = np.random.default_rng(seed)

        # Generate terrain once and cache
        self._terrain: Optional[np.ndarray] = None

    # ── Terrain ────────────────────────────────────────────────────────────────

    def generate_terrain(self) -> np.ndarray:
        """
        Returns (lat, lon, 2) array:
          [:,:,0] elevation in metres (0–3000)
          [:,:,1] coastline distance in grid cells

        Terrain is sampled once and cached on the generator instance.
        Real-data swap: replace with DEM + distance-to-coast raster at
        matching resolution.
        """
        if self._terrain is not None:
            return self._terrain

        H, W = self.grid_size
        # Simple synthetic elevation: smooth random hills
        elev = self.rng.uniform(0, 1, (H, W))
        for _ in range(4):
            elev = np.convolve(elev.ravel(), np.ones(5) / 5, mode='same').reshape(H, W)
        elev = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6) * 3000.0

        # Coast distance: left half is "ocean" (dist increases eastward)
        coast_dist = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                coast_dist[i, j] = float(j)   # simple west-coast proxy

        self._terrain = np.stack([elev, coast_dist], axis=-1)   # (H, W, 2)
        return self._terrain

    # ── Background noise ───────────────────────────────────────────────────────

    def _background(self, shape: Tuple) -> np.ndarray:
        """Per-channel correlated background noise."""
        noise = self.rng.normal(0.0, 1.0, shape)
        return noise

    # ── Gaussian event blob ───────────────────────────────────────────────────

    def _blob(self, grid_shape: Tuple[int, int],
              center: Tuple[float, float],
              radius: float, intensity: float) -> np.ndarray:
        x = np.arange(0, grid_shape[1])
        y = np.arange(0, grid_shape[0])
        xx, yy = np.meshgrid(x, y)
        dist_sq = (xx - center[1]) ** 2 + (yy - center[0]) ** 2
        return intensity * np.exp(-dist_sq / (2 * radius ** 2))

    def _rotating_blob(self, grid_shape: Tuple[int, int],
                       center: Tuple[float, float],
                       radius: float, intensity: float,
                       t: int, clockwise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (u_field, v_field) for a rotating wind pattern."""
        x = np.arange(0, grid_shape[1])
        y = np.arange(0, grid_shape[0])
        xx, yy = np.meshgrid(x, y)
        dx = xx - center[1]
        dy = yy - center[0]
        dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        weight = self._blob(grid_shape, center, radius, intensity)
        sign = -1.0 if clockwise else 1.0
        u = sign * (-dy / dist) * weight
        v = sign * ( dx / dist) * weight
        return u, v

    # ── Channel spike profiles per event type ─────────────────────────────────

    def _apply_event_channels(self,
                              grid: np.ndarray,          # (H, W, C) in-place
                              high_grid: np.ndarray,     # (Hhr, Whr, C) in-place
                              event_type: str,
                              pos: Tuple[float, float],
                              intensity: float,
                              radius: float,
                              t: int):
        H, W, _ = grid.shape
        Hhr, Whr, _ = high_grid.shape
        scale = self.downscale_factor
        pos_hr = (pos[0] * scale, pos[1] * scale)

        if event_type == "extreme_rainfall":
            blob = self._blob((H, W), pos, radius, intensity)
            grid[:, :, CH_RAIN] += blob
            grid[:, :, CH_HUM]  += self._blob((H, W), pos, radius * 1.2, 0.4)
            high_grid[:, :, CH_RAIN] += self._blob((Hhr, Whr), pos_hr, radius * scale, intensity)

        elif event_type == "cyclone":
            # Low pressure + rotating winds + heavy rainfall
            grid[:, :, CH_PRES] -= self._blob((H, W), pos, radius, intensity * 0.5)
            rain = self._blob((H, W), pos, radius * 0.8, intensity * 1.2)
            grid[:, :, CH_RAIN] += rain
            grid[:, :, CH_HUM]  += self._blob((H, W), pos, radius, 0.5)
            u, v = self._rotating_blob((H, W), pos, radius, intensity * 0.6, t)
            grid[:, :, CH_U] += u
            grid[:, :, CH_V] += v
            high_grid[:, :, CH_RAIN] += self._blob((Hhr, Whr), pos_hr,
                                                    radius * scale, intensity * 1.2)
            high_grid[:, :, CH_PRES] -= self._blob((Hhr, Whr), pos_hr,
                                                    radius * scale, intensity * 0.5)

        elif event_type == "heatwave":
            # Sustained temperature anomaly, suppressed humidity, no rain
            grid[:, :, CH_TEMP] += self._blob((H, W), pos, radius * 1.5, intensity * 0.8)
            grid[:, :, CH_HUM]  -= self._blob((H, W), pos, radius, 0.3)
            high_grid[:, :, CH_TEMP] += self._blob((Hhr, Whr), pos_hr,
                                                    radius * scale * 1.5, intensity * 0.8)

        elif event_type == "cold_wave":
            # Sustained negative temperature anomaly
            grid[:, :, CH_TEMP] -= self._blob((H, W), pos, radius * 1.5, intensity * 0.7)
            grid[:, :, CH_PRES] += self._blob((H, W), pos, radius, intensity * 0.3)
            high_grid[:, :, CH_TEMP] -= self._blob((Hhr, Whr), pos_hr,
                                                    radius * scale * 1.5, intensity * 0.7)

    # ── Ground truth helper ───────────────────────────────────────────────────

    def _make_gt(self, event_id: str, event_type: str, t: int,
                 pos: Tuple[float, float], intensity: float,
                 vel: Tuple[float, float], radius: float) -> EventGroundTruth:
        bbox = (pos[0] - radius, pos[1] - radius,
                pos[0] + radius, pos[1] + radius)
        # Synthetic pressure levels: cyclones have deeper vertical extent
        pmin = 850.0 if event_type == "cyclone" else 950.0
        pmax = 1013.0
        return EventGroundTruth(
            event_id=event_id,
            event_type=event_type,
            timestamp=t,
            centroid_lat=pos[0],
            centroid_lon=pos[1],
            intensity=intensity,
            velocity_lat=vel[0],
            velocity_lon=vel[1],
            bbox=bbox,
            pressure_min_hpa=pmin,
            pressure_max_hpa=pmax,
        )

    # ── Scenario generators ───────────────────────────────────────────────────

    def _make_tensors(self):
        C = N_CHANNELS
        lr = np.zeros((self.ensemble_members, self.time_steps,
                       self.grid_size[0], self.grid_size[1], C))
        hr = np.zeros((self.ensemble_members, self.time_steps,
                       self.high_res_grid[0], self.high_res_grid[1], C))
        return lr, hr

    def _fill_background(self, lr, hr, e, t):
        """Fill background noise for all channels."""
        bg_scales = [0.10, 0.20, 0.20, 0.15, 0.05, 0.05]  # per channel
        for c, sc in enumerate(bg_scales):
            lr[e, t, :, :, c] = self._background(self.grid_size) * sc
            hr[e, t, :, :, c] = self._background(self.high_res_grid) * sc

    def generate_scenario_a(self) -> Tuple[np.ndarray, np.ndarray, List[EventGroundTruth]]:
        """
        Scenario A: One moving extreme_rainfall event.
        Returns:
            low_res:  (ensembles, time, lat, lon, 6)
            high_res: (ensembles, time, lat_hr, lon_hr, 6)
            ground_truth: List[EventGroundTruth]
        """
        lr, hr = self._make_tensors()
        gts = []

        start_pos  = (5.0, 5.0)
        velocity   = (0.5, 0.5)
        intensity  = 10.0
        radius     = 3.0
        event_id   = str(uuid.uuid4())

        for e in range(self.ensemble_members):
            e_pos = (start_pos[0] + self.rng.normal(0, 0.5),
                     start_pos[1] + self.rng.normal(0, 0.5))
            e_vel = (velocity[0] + self.rng.normal(0, 0.05),
                     velocity[1] + self.rng.normal(0, 0.05))
            e_int = intensity + self.rng.normal(0, 1.0)

            for t in range(self.time_steps):
                self._fill_background(lr, hr, e, t)
                pos = (e_pos[0] + t * e_vel[0], e_pos[1] + t * e_vel[1])
                self._apply_event_channels(lr[e, t], hr[e, t],
                                           "extreme_rainfall", pos, e_int, radius, t)
                if e == 0:
                    gts.append(self._make_gt(event_id, "extreme_rainfall",
                                             t, pos, e_int, e_vel, radius))
        return lr, hr, gts

    def generate_scenario_b(self) -> Tuple[np.ndarray, np.ndarray, List[EventGroundTruth]]:
        """
        Scenario B: Two simultaneous events approaching each other.
        """
        lr, hr = self._make_tensors()
        gts = []

        e1_start = (5.0, 5.0);   e1_vel = (0.3, 0.3);   e1_int = 10.0; e1_r = 3.0
        e2_start = (25.0, 25.0); e2_vel = (-0.3, -0.3); e2_int = 12.0; e2_r = 3.0
        e1_id, e2_id = str(uuid.uuid4()), str(uuid.uuid4())

        for e in range(self.ensemble_members):
            for t in range(self.time_steps):
                self._fill_background(lr, hr, e, t)
                p1 = (e1_start[0] + t * e1_vel[0], e1_start[1] + t * e1_vel[1])
                p2 = (e2_start[0] + t * e2_vel[0], e2_start[1] + t * e2_vel[1])
                self._apply_event_channels(lr[e, t], hr[e, t],
                                           "extreme_rainfall", p1, e1_int, e1_r, t)
                self._apply_event_channels(lr[e, t], hr[e, t],
                                           "extreme_rainfall", p2, e2_int, e2_r, t)
                if e == 0:
                    gts.append(self._make_gt(e1_id, "extreme_rainfall",
                                             t, p1, e1_int, e1_vel, e1_r))
                    gts.append(self._make_gt(e2_id, "extreme_rainfall",
                                             t, p2, e2_int, e2_vel, e2_r))
        return lr, hr, gts

    def generate_cyclone_scenario(self) -> Tuple[np.ndarray, np.ndarray, List[EventGroundTruth]]:
        """
        Scripted cyclone scenario (stand-in for Cyclone-Amphan-like track):
        - Northwest-moving low-pressure system with rotating winds + heavy rainfall.
        - Used for §8 historical replay validation.
        """
        lr, hr = self._make_tensors()
        gts = []

        # Starts in southeast quadrant, moves northwest (like Bay of Bengal cyclones)
        start_pos = (24.0, 26.0)
        velocity  = (-0.4, -0.35)
        intensity = 14.0
        radius    = 4.0
        event_id  = str(uuid.uuid4())

        for e in range(self.ensemble_members):
            e_pos = (start_pos[0] + self.rng.normal(0, 0.3),
                     start_pos[1] + self.rng.normal(0, 0.3))
            e_vel = (velocity[0] + self.rng.normal(0, 0.03),
                     velocity[1] + self.rng.normal(0, 0.03))
            e_int = intensity + self.rng.normal(0, 1.5)

            for t in range(self.time_steps):
                self._fill_background(lr, hr, e, t)
                pos = (e_pos[0] + t * e_vel[0], e_pos[1] + t * e_vel[1])
                self._apply_event_channels(lr[e, t], hr[e, t],
                                           "cyclone", pos, e_int, radius, t)
                if e == 0:
                    gts.append(self._make_gt(event_id, "cyclone",
                                             t, pos, e_int, e_vel, radius))
        return lr, hr, gts

    def generate_heatwave_scenario(self) -> Tuple[np.ndarray, np.ndarray, List[EventGroundTruth]]:
        """
        Scripted heatwave scenario (land-locked temperature anomaly):
        - Sustained +temp anomaly, suppressed humidity, stationary.
        - Used for §8 historical replay validation.
        """
        lr, hr = self._make_tensors()
        gts = []

        # Stationary heat dome over central grid
        start_pos = (16.0, 16.0)
        velocity  = (0.05, 0.02)   # near-stationary
        intensity = 11.0
        radius    = 5.0
        event_id  = str(uuid.uuid4())

        for e in range(self.ensemble_members):
            e_pos = (start_pos[0] + self.rng.normal(0, 0.2),
                     start_pos[1] + self.rng.normal(0, 0.2))
            e_vel = (velocity[0] + self.rng.normal(0, 0.01),
                     velocity[1] + self.rng.normal(0, 0.01))
            e_int = intensity + self.rng.normal(0, 0.8)

            for t in range(self.time_steps):
                self._fill_background(lr, hr, e, t)
                pos = (e_pos[0] + t * e_vel[0], e_pos[1] + t * e_vel[1])
                self._apply_event_channels(lr[e, t], hr[e, t],
                                           "heatwave", pos, e_int, radius, t)
                if e == 0:
                    gts.append(self._make_gt(event_id, "heatwave",
                                             t, pos, e_int, e_vel, radius))
        return lr, hr, gts

    # ── Derived-field helpers (NOT extra channels) ────────────────────────────

    @staticmethod
    def compute_wind_speed(data: np.ndarray) -> np.ndarray:
        """data[..., 1:3] = (u, v); returns wind speed array same spatial shape."""
        return np.sqrt(data[..., CH_U] ** 2 + data[..., CH_V] ** 2)

    @staticmethod
    def compute_moisture_convergence(data: np.ndarray) -> np.ndarray:
        """
        Proxy: –humidity × divergence(u, v).
        Positive value = convergence (moisture influx → rainfall likely).
        """
        u = data[..., CH_U]
        v = data[..., CH_V]
        hum = data[..., CH_HUM]
        du_dx = np.gradient(u, axis=-1)   # lon axis (last spatial)
        dv_dy = np.gradient(v, axis=-2)   # lat axis
        divergence = du_dx + dv_dy
        return -hum * divergence           # convergence = negative divergence


# ── Data loader interface ──────────────────────────────────────────────────────

class SyntheticDataLoader:
    """
    Clean interface — mirrors the shape of a RealWeatherDataLoader.
    Real-data swap: subclass and override load_data() to parse NetCDF/Xarray.
    """

    _SCENARIO_MAP = {
        "A":        "generate_scenario_a",
        "B":        "generate_scenario_b",
        "cyclone":  "generate_cyclone_scenario",
        "heatwave": "generate_heatwave_scenario",
    }

    def __init__(self, scenario: str = "A", seed: Optional[int] = None):
        self.generator = SyntheticWeatherGenerator(seed=seed)
        self.scenario = scenario

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[EventGroundTruth]]:
        method = self._SCENARIO_MAP.get(self.scenario)
        if method is None:
            raise NotImplementedError(
                f"Scenario '{self.scenario}' not implemented. "
                f"Available: {list(self._SCENARIO_MAP)}"
            )
        return getattr(self.generator, method)()

    def get_terrain(self) -> np.ndarray:
        """Returns (lat, lon, 2) terrain array: [elevation, coastline_distance]."""
        return self.generator.generate_terrain()


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for scen in ["A", "B", "cyclone", "heatwave"]:
        loader = SyntheticDataLoader(scenario=scen, seed=42)
        lr, hr, gt = loader.load_data()
        print(f"Scenario {scen}: low_res={lr.shape}, high_res={hr.shape}, "
              f"gt={len(gt)}, event_type={gt[0].event_type}")
    terrain = loader.get_terrain()
    print(f"Terrain shape: {terrain.shape}")
    ws = SyntheticWeatherGenerator.compute_wind_speed(lr[0, 0])
    mc = SyntheticWeatherGenerator.compute_moisture_convergence(lr[0, 0])
    print(f"Wind speed max: {ws.max():.3f}, Moisture conv max: {mc.max():.3f}")
