"""
run_pipeline.py — Main Entry Point
====================================
Runs the full Extreme Weather Monitoring pipeline.

Two paths:
  1. ST-GNN path (default, new)      — SpatioTemporalGNN → 4D BBox → risk output
  2. Baseline path (comparator only) — percentile threshold → SimpleTracker

Usage:
    python run_pipeline.py              # runs both scenarios (A and B)
    python run_pipeline.py cyclone      # runs cyclone scenario

API usage (return_results=True):
    from run_pipeline import run_pipeline
    result = run_pipeline(scenario="cyclone", return_results=True)
    # result is a dict with keys: events, metrics, uncertainty, tracks

Real-data swap point:
    Replace SyntheticDataLoader with RealWeatherDataLoader that parses
    xarray/NetCDF NCMRWF NEPS-G files. Everything downstream is unchanged.
"""
import numpy as np
import sys
import os
from typing import Optional

from src.data.synthetic_generator import SyntheticDataLoader
from src.data.climatology import SyntheticClimatology
from src.data.efi import EFIScorer
from src.models.baseline_detector import BaselineDetector
from src.models.tracker import SimpleTracker, TrackedEvent
from src.models.gnn_tracker import GNNTracker        # baseline GNN (old)
from src.models.st_gnn import SpatioTemporalEventDetector
from src.models.ensemble_uncertainty import EnsembleUncertaintyEstimator
from src.models.physics_loss import physics_plausibility_score
from src.evaluation.metrics import Evaluator


# ── ST-GNN evaluation helper ───────────────────────────────────────────────────

def _evaluate_stgnn_events(events, gt_list):
    """
    Evaluate ST-GNN 4D bounding boxes against ground truth.
    Computes bbox IoU, centroid error, event detection rate, and event-type accuracy.
    """
    if not events or not gt_list:
        return {
            "detection_rate": 0.0,
            "mean_bbox_iou": 0.0,
            "mean_centroid_error": float("inf"),
            "n_detected": 0,
        }

    from src.evaluation.metrics import compute_iou as _iou
    # Collect GT centroids and bboxes per unique event_id
    gt_by_id = {}
    for g in gt_list:
        if g.event_id not in gt_by_id:
            gt_by_id[g.event_id] = g

    ious, c_errors = [], []
    for g in gt_by_id.values():
        best_iou  = 0.0
        best_cerr = float("inf")
        for ev in events:
            lat_r = ev.get("latitude_range", [0, 1])
            lon_r = ev.get("longitude_range", [0, 1])
            ev_box = (lat_r[0], lon_r[0], lat_r[1], lon_r[1])
            gt_box = g.bbox
            iou = _iou(ev_box, gt_box)
            c_lat = (lat_r[0] + lat_r[1]) / 2
            c_lon = (lon_r[0] + lon_r[1]) / 2
            cerr = ((c_lat - g.centroid_lat)**2 + (c_lon - g.centroid_lon)**2) ** 0.5
            if iou > best_iou:
                best_iou  = iou
                best_cerr = cerr
        ious.append(best_iou)
        c_errors.append(best_cerr)

    det_rate = sum(1 for iou in ious if iou > 0.0) / max(len(ious), 1)
    return {
        "detection_rate":       round(det_rate, 4),
        "mean_bbox_iou":        round(float(np.mean(ious)),      4),
        "mean_centroid_error":  round(float(np.mean(c_errors)),  4),
        "n_detected":           len(events),
    }


# ── ST-GNN pipeline ────────────────────────────────────────────────────────────

def run_pipeline(scenario: str = "B",
                 return_results: bool = False,
                 verbose: bool = True) -> Optional[dict]:
    """
    Run the full ST-GNN-based pipeline for the given scenario.

    Args:
        scenario:       A / B / cyclone / heatwave
        return_results: If True, return a results dict (used by the API).
        verbose:        Print progress.

    Returns:
        If return_results=True → dict with keys: events, metrics, uncertainty, tracks
        Otherwise → None
    """
    _pr = print if verbose else (lambda *a, **k: None)
    _pr(f"\n{'='*60}")
    _pr(f"  ST-GNN PIPELINE — Scenario {scenario}")
    _pr(f"{'='*60}")

    # ── 1. Data ────────────────────────────────────────────────────────────────
    _pr("1. Loading synthetic data...")
    loader   = SyntheticDataLoader(scenario=scenario, seed=0)
    lr, hr, gt = loader.load_data()
    terrain  = loader.get_terrain()
    _pr(f"   Input: {lr.shape}  (ensemble, time, lat, lon, channels)")
    _pr(f"   Terrain: {terrain.shape}")

    # ── 2. Climatology + EFI ──────────────────────────────────────────────────
    _pr("2. Building climatological baseline + EFI scores...")
    clim = SyntheticClimatology()
    clim.generate_historical_baseline()

    scorer = EFIScorer(clim, threshold=0.5)
    # Time-mean EFI across timesteps (use for detector), per ensemble member
    efi_per_member = np.stack([
        scorer.compute(lr[:, t, :, :, 0], feature_idx=0)
        for t in range(lr.shape[1])
    ], axis=0).mean(axis=0)   # (H, W)
    efi_ensemble = np.stack([efi_per_member] * lr.shape[0], axis=0)  # (E, H, W)
    _pr(f"   EFI range: [{efi_per_member.min():.3f}, {efi_per_member.max():.3f}]")

    # ── 3. Physics plausibility ────────────────────────────────────────────────
    _pr("3. Computing physics plausibility score...")
    mean_frame = lr[:, lr.shape[1]//2].mean(axis=0)  # mid-forecast, ensemble mean
    phys_score = physics_plausibility_score(
        rainfall  = mean_frame[:, :, 0],
        wind_u    = mean_frame[:, :, 1],
        wind_v    = mean_frame[:, :, 2],
        humidity  = mean_frame[:, :, 5],
    )
    _pr(f"   Physics plausibility: {phys_score:.3f}")

    # ── 4. ST-GNN detection ────────────────────────────────────────────────────
    _pr("4. Running SpatioTemporalGNN detection + 4D bounding box extraction...")
    detector = SpatioTemporalEventDetector(
        grid_h=lr.shape[2],
        grid_w=lr.shape[3],
        detection_threshold=0.25,
        efi_threshold=0.5,
    )
    events = detector.detect(
        data=lr,
        terrain=terrain,
        efi_grids=efi_ensemble,
        forecast_lead=24,
        physics_plausibility=phys_score,
    )
    _pr(f"   Detected {len(events)} events:")
    for ev in events[:5]:
        _pr(f"     [{ev['event_type']:18s}] sev={ev['severity']:.2f}  "
            f"conf={ev['confidence']:.2f}  risk={ev['risk_level']:8s}  "
            f"lat={ev['latitude_range']}  t={ev['t_start']}..{ev['t_end']}")

    # ── 5. TrackedEvent conversion for API / downstream ────────────────────────
    tracked = [TrackedEvent.from_bbox_dict(ev, start_time=ev.get("t_start", 0))
               for ev in events]

    # ── 6. Ensemble uncertainty ────────────────────────────────────────────────
    _pr("6. Ensemble uncertainty estimation...")
    all_ensemble_tracks = {}
    for e in range(lr.shape[0]):
        base_tracker = SimpleTracker(max_distance=5.0)
        detector_base = BaselineDetector(clim, feature_idx=0, percentile=95)
        for t in range(lr.shape[1]):
            objects = detector_base.detect(lr[e, t], time_step=t)
            base_tracker.update(objects, time_step=t)
        all_ensemble_tracks[e] = base_tracker.get_all_tracks()

    uncertainty = EnsembleUncertaintyEstimator().estimate_uncertainty(
        all_ensemble_tracks, target_event_idx=0
    )
    for k, v in uncertainty.items():
        _pr(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    # ── 7. Evaluation: ST-GNN bbox IoU against GT bboxes ─────────────────────
    _pr("7. Evaluating ST-GNN 4D boxes against ground truth...")
    metrics = _evaluate_stgnn_events(events, gt)
    for k, v in metrics.items():
        _pr(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    _pr(f"\n  [OK] Pipeline complete for scenario '{scenario}'")

    if return_results:
        return {
            "events":      [dict(ev) for ev in events],
            "tracked":     tracked,
            "metrics":     metrics,
            "uncertainty": uncertainty,
            "physics_plausibility": phys_score,
        }


# ── Baseline pipeline (comparator only) ───────────────────────────────────────

def run_baseline_pipeline(scenario: str = "B", verbose: bool = True):
    """
    Original detect->track pipeline kept as baseline comparator.
    Referenced in README section 9.
    """
    _pr = print if verbose else (lambda *a, **k: None)
    _pr(f"\n--- BASELINE PIPELINE -- Scenario {scenario} ---")

    loader = SyntheticDataLoader(scenario=scenario)
    lr, _, gt = loader.load_data()
    clim = SyntheticClimatology()
    clim.generate_historical_baseline()
    detector = BaselineDetector(clim, feature_idx=0, percentile=95)

    model_path = "gnn_tracker.pth"
    use_gnn = os.path.exists(model_path)
    _pr(f"   Tracker: {'GNN' if use_gnn else 'Simple Euclidean'}")

    all_ensemble_tracks = {}
    for e in range(lr.shape[0]):
        tracker = GNNTracker(model_path) if use_gnn else SimpleTracker(5.0)
        for t in range(lr.shape[1]):
            objects = detector.detect(lr[e, t], time_step=t)
            tracker.update(objects, time_step=t)
        all_ensemble_tracks[e] = tracker.get_all_tracks()

    valid_tracks = [tr for tr in all_ensemble_tracks[0] if tr.duration >= 5]
    _pr(f"   Found {len(valid_tracks)} significant tracks (duration >= 5)")

    uncertainty = EnsembleUncertaintyEstimator().estimate_uncertainty(
        all_ensemble_tracks, 0
    )
    evaluator = Evaluator(gt, valid_tracks)
    metrics   = evaluator.evaluate_localization()
    for k, v in metrics.items():
        _pr(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    return {"tracks": valid_tracks, "metrics": metrics, "uncertainty": uncertainty}


if __name__ == "__main__":
    scenarios = sys.argv[1:] if len(sys.argv) > 1 else ["A", "B"]
    for scen in scenarios:
        run_pipeline(scenario=scen)
        print()
        run_baseline_pipeline(scenario=scen)
        print("=" * 60)
