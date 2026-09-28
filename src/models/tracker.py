"""
tracker.py — Event Tracking State + 4D Schema
==============================================
TrackedEvent now carries the full output schema required by §5:

    event_id, event_type, forecast_time,
    latitude_range, longitude_range, vertical_range,
    severity (0-1), confidence (0-1),
    probability_exceedance, physics_plausibility

Plus:
    risk_level property — low/moderate/severe/extreme
    to_api_dict()       — serialisation for FastAPI

SimpleTracker (baseline Euclidean tracker) is unchanged.
The new extended fields default to None so existing code continues working.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import math
import uuid
from dataclasses import dataclass, field

try:
    from .baseline_detector import DetectedObject
except ImportError:
    from baseline_detector import DetectedObject


@dataclass
class TrackedEvent:
    # ── Core identity (unchanged) ──────────────────────────────────────────────
    event_id:   str
    start_time: int
    end_time:   int
    history:    List[DetectedObject]

    # ── Extended 4D schema (§5) ── all Optional so existing callers pass None ──
    event_type:            Optional[str]   = None   # cyclone/heatwave/…/normal
    forecast_time:         Optional[int]   = None   # lead-time hours
    # Spatial bounding box
    latitude_range:        Optional[Tuple[float, float]] = None   # (min, max)
    longitude_range:       Optional[Tuple[float, float]] = None   # (min, max)
    # Vertical (4th spatial dimension)
    vertical_range:        Optional[Tuple[float, float]] = None   # (hPa_min, hPa_max)
    # Probabilistic
    severity:              Optional[float] = None   # 0–1
    confidence_score:      Optional[float] = None   # 0–1
    probability_exceedance:Optional[float] = None   # 0–1
    physics_plausibility:  Optional[float] = None   # 0–1

    # ── Derived properties (original, unchanged) ───────────────────────────────

    @property
    def current_lat(self) -> float:
        return self.history[-1].centroid_lat

    @property
    def current_lon(self) -> float:
        return self.history[-1].centroid_lon

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time + 1

    @property
    def speed_and_direction(self) -> tuple:
        """Returns (speed, direction_deg) from last 2 steps if available."""
        if len(self.history) < 2:
            return 0.0, 0.0
        p1, p2 = self.history[-2], self.history[-1]
        dlat = p2.centroid_lat - p1.centroid_lat
        dlon = p2.centroid_lon - p1.centroid_lon
        speed     = math.sqrt(dlat**2 + dlon**2)
        direction = math.degrees(math.atan2(dlat, dlon))
        return speed, direction

    # ── New: risk level ────────────────────────────────────────────────────────

    @property
    def risk_level(self) -> str:
        """
        Combines severity × confidence into a risk tier.

        Rule: a low-confidence severe event is capped at 'moderate'
        to prevent false high-priority alerts (matches brief requirement).

        Thresholds:
            severity × confidence < 0.20 → low
            0.20–0.50                    → moderate
            0.50–0.75                    → severe
            > 0.75                       → extreme
        """
        sev  = self.severity       or 0.0
        conf = self.confidence_score or 0.0
        # Cap rule: high-severity, low-confidence → moderate
        if conf < 0.3 and sev > 0.7:
            return "moderate"
        score = sev * conf
        if score < 0.20:
            return "low"
        elif score < 0.50:
            return "moderate"
        elif score < 0.75:
            return "severe"
        else:
            return "extreme"

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_api_dict(self) -> Dict:
        """
        Serialise to the full API schema (§5 + §6).
        All fields are JSON-serialisable (no numpy types).
        """
        speed, direction = self.speed_and_direction
        return {
            "event_id":               self.event_id,
            "event_type":             self.event_type or "unknown",
            "forecast_time":          self.forecast_time or self.start_time,
            "start_time":             self.start_time,
            "end_time":               self.end_time,
            "duration":               self.duration,
            "latitude_range":         list(self.latitude_range)  if self.latitude_range  else
                                      [self.current_lat - 3, self.current_lat + 3],
            "longitude_range":        list(self.longitude_range) if self.longitude_range else
                                      [self.current_lon - 3, self.current_lon + 3],
            "vertical_range":         list(self.vertical_range)  if self.vertical_range  else
                                      [700.0, 1013.0],
            "severity":               round(float(self.severity or 0.0), 4),
            "confidence":             round(float(self.confidence_score or 0.0), 4),
            "probability_exceedance": round(float(self.probability_exceedance or 0.0), 4),
            "physics_plausibility":   round(float(self.physics_plausibility or 1.0), 4),
            "risk_level":             self.risk_level,
            "speed":                  round(speed, 4),
            "direction_deg":          round(direction, 2),
            "centroid_lat":           round(self.current_lat, 4),
            "centroid_lon":           round(self.current_lon, 4),
        }

    @classmethod
    def from_bbox_dict(cls, bbox: Dict, start_time: int,
                       detected_obj: Optional["DetectedObject"] = None) -> "TrackedEvent":
        """
        Construct a TrackedEvent from an Event4DBoundingBox dict (ST-GNN output).
        """
        # Create a synthetic DetectedObject for history if not provided
        if detected_obj is None:
            lat_r = bbox.get("latitude_range",  [0, 1])
            lon_r = bbox.get("longitude_range", [0, 1])
            detected_obj = DetectedObject(
                object_id=bbox["event_id"],
                timestamp=start_time,
                centroid_lat=(lat_r[0] + lat_r[1]) / 2,
                centroid_lon=(lon_r[0] + lon_r[1]) / 2,
                bbox=(lat_r[0], lon_r[0], lat_r[1], lon_r[1]),
                area=int((lat_r[1]-lat_r[0]) * (lon_r[1]-lon_r[0])),
                intensity=float(bbox.get("severity", 0.5)) * 20.0,
            )
        return cls(
            event_id=bbox["event_id"],
            start_time=start_time,
            end_time=bbox.get("t_end", start_time),
            history=[detected_obj],
            event_type=bbox.get("event_type"),
            forecast_time=bbox.get("forecast_time"),
            latitude_range=tuple(bbox["latitude_range"])  if bbox.get("latitude_range")  else None,
            longitude_range=tuple(bbox["longitude_range"]) if bbox.get("longitude_range") else None,
            vertical_range=tuple(bbox["vertical_range"])  if bbox.get("vertical_range")  else None,
            severity=bbox.get("severity"),
            confidence_score=bbox.get("confidence"),
            probability_exceedance=bbox.get("probability_exceedance"),
            physics_plausibility=bbox.get("physics_plausibility"),
        )


# ── Baseline tracker (unchanged logic) ────────────────────────────────────────

class SimpleTracker:
    """
    Baseline Euclidean-distance tracker.
    Logic is unchanged from the original — kept as comparator.
    """
    def __init__(self, max_distance: float = 5.0):
        self.max_distance    = max_distance
        self.active_tracks:  List[TrackedEvent] = []
        self.finished_tracks:List[TrackedEvent] = []

    def update(self, detected_objects: List[DetectedObject], time_step: int):
        if not self.active_tracks:
            for obj in detected_objects:
                self.active_tracks.append(TrackedEvent(
                    event_id=str(uuid.uuid4()),
                    start_time=time_step,
                    end_time=time_step,
                    history=[obj]
                ))
            return

        unassigned_objects = list(detected_objects)
        next_active_tracks = []

        for track in self.active_tracks:
            best_obj  = None
            best_dist = float("inf")
            for obj in unassigned_objects:
                dist = math.sqrt((track.current_lat - obj.centroid_lat)**2 +
                                 (track.current_lon - obj.centroid_lon)**2)
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_obj  = obj
            if best_obj:
                track.history.append(best_obj)
                track.end_time = time_step
                next_active_tracks.append(track)
                unassigned_objects.remove(best_obj)
            else:
                self.finished_tracks.append(track)

        for obj in unassigned_objects:
            next_active_tracks.append(TrackedEvent(
                event_id=str(uuid.uuid4()),
                start_time=time_step,
                end_time=time_step,
                history=[obj]
            ))
        self.active_tracks = next_active_tracks

    def get_all_tracks(self) -> List[TrackedEvent]:
        return self.finished_tracks + self.active_tracks


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    from data.synthetic_generator import SyntheticDataLoader
    from data.climatology import SyntheticClimatology
    from models.baseline_detector import BaselineDetector

    loader = SyntheticDataLoader(scenario="A", seed=0)
    lr, _, gt = loader.load_data()
    clim = SyntheticClimatology()
    clim.generate_historical_baseline()
    detector = BaselineDetector(clim, feature_idx=0, percentile=95)
    tracker  = SimpleTracker(max_distance=3.0)

    for t in range(lr.shape[1]):
        objects = detector.detect(lr[0, t], time_step=t)
        tracker.update(objects, time_step=t)

    tracks = tracker.get_all_tracks()
    print(f"Tracks: {len(tracks)}")
    if tracks:
        longest = max(tracks, key=lambda x: x.duration)
        # Patch in some §5 fields to test serialisation
        longest.event_type    = "extreme_rainfall"
        longest.severity      = 0.8
        longest.confidence_score = 0.7
        longest.vertical_range = (700.0, 1013.0)
        print(f"Risk level: {longest.risk_level}")
        d = longest.to_api_dict()
        print("API dict keys:", list(d.keys()))
