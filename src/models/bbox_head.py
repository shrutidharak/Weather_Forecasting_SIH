"""
bbox_head.py — 4D Bounding Box Head
=====================================
Extracts Event4DBoundingBox objects from node-level detection scores
and event-type probability maps produced by SpatioTemporalGNN.

Schema (matches §5 spec exactly):
    event_id:              str   — UUID
    event_type:            str   — cyclone/heatwave/cold_wave/extreme_rainfall/normal
    forecast_time:         int   — lead time in hours
    latitude_range:        tuple — (lat_min, lat_max) in grid indices
    longitude_range:       tuple — (lon_min, lon_max) in grid indices
    vertical_range:        tuple — (pressure_min_hPa, pressure_max_hPa) synthetic
    severity:              float — 0–1 (normalised from detection score)
    confidence:            float — 0–1 (ensemble spread → confidence)
    probability_exceedance:float — fraction of ensemble exceeding threshold
    physics_plausibility:  float — from PhysicsLoss module
    risk_level:            str   — low/moderate/severe/extreme

4D bounding box dimensions:
    1. lat_min / lat_max   → spatial
    2. lon_min / lon_max   → spatial
    3. pressure_min / max  → vertical (proxy from terrain + event type)
    4. t_start / t_end     → temporal (derived from detection mask over time)
"""
import numpy as np
import uuid
from typing import List, Tuple, Dict, Any, Optional
from scipy.ndimage import label, find_objects, center_of_mass

# ── Event type catalogue ───────────────────────────────────────────────────────
EVENT_TYPE_NAMES = ["cyclone", "heatwave", "cold_wave", "extreme_rainfall", "normal"]

# Synthetic vertical extents per event type (hPa)
_VERTICAL_EXTENTS = {
    "cyclone":          (500,  1013),
    "heatwave":         (850,  1013),
    "cold_wave":        (700,  1013),
    "extreme_rainfall": (700,  1013),
    "normal":           (1000, 1013),
}

# Risk-level mapping (severity × confidence)
def _risk_level(severity: float, confidence: float) -> str:
    """
    Rule: low-confidence severe events are capped at 'moderate'
    to avoid high-severity false alerts.
    """
    score = severity * confidence
    if confidence < 0.3 and severity > 0.7:
        return "moderate"    # downgrade high-severity, low-confidence events
    if score < 0.2:
        return "low"
    elif score < 0.5:
        return "moderate"
    elif score < 0.75:
        return "severe"
    else:
        return "extreme"


# ── TypedDict-style output schema ──────────────────────────────────────────────
# (Using plain dict for Python 3.8 compat; TypedDict imported below for editors)
try:
    from typing import TypedDict

    class Event4DBoundingBox(TypedDict):
        event_id:               str
        event_type:             str
        forecast_time:          int
        latitude_range:         Tuple[float, float]
        longitude_range:        Tuple[float, float]
        vertical_range:         Tuple[float, float]
        t_start:                int
        t_end:                  int
        severity:               float
        confidence:             float
        probability_exceedance: float
        physics_plausibility:   float
        risk_level:             str

except ImportError:
    Event4DBoundingBox = dict


def _make_event(
    event_type:             str,
    lat_min: float,  lat_max: float,
    lon_min: float,  lon_max: float,
    t_start: int,    t_end:   int,
    severity:               float,
    confidence:             float,
    probability_exceedance: float,
    physics_plausibility:   float,
    forecast_time:          int,
) -> Dict[str, Any]:
    pmin, pmax = _VERTICAL_EXTENTS.get(event_type, (700, 1013))
    return {
        "event_id":               str(uuid.uuid4()),
        "event_type":             event_type,
        "forecast_time":          forecast_time,
        "latitude_range":         (round(lat_min, 2), round(lat_max, 2)),
        "longitude_range":        (round(lon_min, 2), round(lon_max, 2)),
        "vertical_range":         (float(pmin), float(pmax)),
        "t_start":                t_start,
        "t_end":                  t_end,
        "severity":               round(float(np.clip(severity,   0, 1)), 4),
        "confidence":             round(float(np.clip(confidence, 0, 1)), 4),
        "probability_exceedance": round(float(np.clip(probability_exceedance, 0, 1)), 4),
        "physics_plausibility":   round(float(np.clip(physics_plausibility,   0, 1)), 4),
        "risk_level":             _risk_level(severity, confidence),
    }


class BoundingBoxHead:
    """
    Post-processes ST-GNN node-level outputs into Event4DBoundingBox objects.

    Steps:
    1. Threshold detection_scores → binary event mask (per timestep)
    2. Connected-components over the time-mean mask → spatial blobs
    3. For each blob: determine event_type (argmax of mean et_probs in blob)
    4. Compute 4D extents (lat/lon from blob, vertical from event type,
       temporal from first/last timestep the blob is active)
    5. Severity = peak detection score in blob
    """

    def __init__(self, grid_h: int, grid_w: int):
        self.grid_h = grid_h
        self.grid_w = grid_w

    def extract_events(
        self,
        detection_scores:       "torch.Tensor",  # (T, N) in [0,1]
        event_type_probs:       "torch.Tensor",  # (T, N, 5)
        detection_threshold:    float = 0.3,
        terrain:                Optional[np.ndarray] = None,  # (H,W,2)
        raw_data:               Optional[np.ndarray] = None,  # (T, H, W, C)
        forecast_lead:          int = 0,
        ensemble_confidence:    float = 0.5,
        physics_plausibility:   float = 1.0,
    ) -> List[Dict[str, Any]]:
        import torch
        T  = detection_scores.shape[0]
        N  = detection_scores.shape[1]
        H, W = self.grid_h, self.grid_w

        det_np = detection_scores.numpy()         # (T, N)
        et_np  = event_type_probs.numpy()         # (T, N, 5)

        # Reshape to spatial grids
        det_grid = det_np.reshape(T, H, W)        # (T, H, W)
        et_grid  = et_np.reshape(T, H, W, 5)      # (T, H, W, 5)

        # Time-mean detection → connected component labelling
        mean_det = det_grid.mean(axis=0)           # (H, W)
        binary   = (mean_det > detection_threshold).astype(np.int32)

        labeled, n_blobs = label(binary)
        if n_blobs == 0:
            return []

        events = []
        slices = find_objects(labeled)
        centroids = center_of_mass(mean_det, labeled, list(range(1, n_blobs + 1)))
        if isinstance(centroids, tuple):
            centroids = [centroids]

        for blob_idx, sl in enumerate(slices):
            lat_sl, lon_sl = sl
            blob_mask = (labeled == (blob_idx + 1))   # (H, W) bool

            # Event type: argmax of mean probability inside blob
            et_in_blob = et_grid[:, blob_mask, :]    # (T, n_cells, 5)
            mean_et    = et_in_blob.mean(axis=(0, 1)) # (5,)
            et_idx     = int(np.argmax(mean_et))
            event_type = EVENT_TYPE_NAMES[et_idx]

            # Physical signal check: if classification is normal or to refine event_type
            if raw_data is not None:
                # raw_data: (T, H, W, C) where C: 0=rain, 1=u, 2=v, 3=temp, 4=pres, 5=hum
                rain_val = float(raw_data[:, blob_mask, 0].max())
                temp_val = float(raw_data[:, blob_mask, 3].mean())
                pres_val = float(raw_data[:, blob_mask, 4].min())
                if pres_val < -1.5 and rain_val > 2.0:
                    event_type = "cyclone"
                elif temp_val > 2.0:
                    event_type = "heatwave"
                elif temp_val < -2.0:
                    event_type = "cold_wave"
                elif rain_val > 4.0:
                    event_type = "extreme_rainfall"

            # Skip normal detections
            if event_type == "normal" and mean_et[et_idx] > 0.8:
                continue

            # Temporal extent: first and last T with any active detection
            det_blob_t = det_grid[:, blob_mask].mean(axis=1)  # (T,)
            active_t   = np.where(det_blob_t > detection_threshold)[0]
            t_start    = int(active_t[0])  if len(active_t) > 0 else 0
            t_end      = int(active_t[-1]) if len(active_t) > 0 else T - 1

            # Severity = peak detection score in blob
            severity = float(det_grid[:, blob_mask].max())

            # Spatial extents
            lat_min = float(lat_sl.start)
            lat_max = float(lat_sl.stop - 1)
            lon_min = float(lon_sl.start)
            lon_max = float(lon_sl.stop - 1)

            # Probability of exceedance (use ensemble confidence as proxy here)
            prob_exceed = float(ensemble_confidence)

            ev = _make_event(
                event_type=event_type,
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                t_start=t_start, t_end=t_end,
                severity=severity,
                confidence=ensemble_confidence,
                probability_exceedance=prob_exceed,
                physics_plausibility=physics_plausibility,
                forecast_time=forecast_lead,
            )
            events.append(ev)

        return events


if __name__ == "__main__":
    print("Event type catalogue:", EVENT_TYPE_NAMES)
    # Quick risk-level sanity checks
    print("high-sev, low-conf → moderate:", _risk_level(0.9, 0.2))  # capped
    print("high-sev, high-conf → extreme:", _risk_level(0.9, 0.9))
    print("low-sev, low-conf → low:",       _risk_level(0.1, 0.1))
