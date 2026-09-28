import numpy as np
from typing import List
import math
import sys
import os

try:
    from .data.synthetic_generator import EventGroundTruth
    from .models.tracker import TrackedEvent
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    from data.synthetic_generator import EventGroundTruth
    from models.tracker import TrackedEvent

def compute_iou(boxA, boxB):
    # box: (min_lat, min_lon, max_lat, max_lon)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class Evaluator:
    def __init__(self, ground_truths: List[EventGroundTruth], predicted_tracks: List[TrackedEvent]):
        self.ground_truths = ground_truths
        self.predicted_tracks = predicted_tracks
        
        
    def match_tracks(self) -> dict:
        """
        Matches predicted tracks to ground truth based on longest temporal overlap and spatial proximity.
        Returns a dict mapping gt_event_id to the best predicted track.
        """
        gt_by_id = {}
        for gt in self.ground_truths:
            if gt.event_id not in gt_by_id:
                gt_by_id[gt.event_id] = []
            gt_by_id[gt.event_id].append(gt)
            
        matches = {}
        for gt_id, gt_list in gt_by_id.items():
            best_track = None
            min_mean_error = float('inf')
            
            for track in self.predicted_tracks:
                errors = []
                for gt in gt_list:
                    pred_obj = next((obj for obj in track.history if obj.timestamp == gt.timestamp), None)
                    if pred_obj:
                        dist = math.sqrt((pred_obj.centroid_lat - gt.centroid_lat)**2 + (pred_obj.centroid_lon - gt.centroid_lon)**2)
                        errors.append(dist)
                
                if len(errors) > 0:
                    mean_error = np.mean(errors)
                    # We want the track that has the lowest mean centroid error AND covers a reasonable amount of the sequence
                    if mean_error < min_mean_error and len(errors) > 5:
                        min_mean_error = mean_error
                        best_track = track
                        
            matches[gt_id] = best_track
            
        return matches

    def evaluate_localization(self):
        matches = self.match_tracks()
        if not matches:
            return {"error": "No tracks matched"}
            
        all_centroid_errors = []
        all_ious = []
        all_intensity_errors = []
        total_gt_steps = 0
        total_matched_steps = 0
        
        gt_by_id = {}
        for gt in self.ground_truths:
            if gt.event_id not in gt_by_id:
                gt_by_id[gt.event_id] = []
            gt_by_id[gt.event_id].append(gt)
            
        for gt_id, gt_list in gt_by_id.items():
            total_gt_steps += len(gt_list)
            pred_track = matches.get(gt_id)
            
            if not pred_track:
                continue
                
            for gt in gt_list:
                pred_obj = next((obj for obj in pred_track.history if obj.timestamp == gt.timestamp), None)
                if pred_obj:
                    error = math.sqrt((pred_obj.centroid_lat - gt.centroid_lat)**2 + (pred_obj.centroid_lon - gt.centroid_lon)**2)
                    all_centroid_errors.append(error)
                    
                    iou = compute_iou(pred_obj.bbox, gt.bbox)
                    all_ious.append(iou)
                    
                    all_intensity_errors.append(abs(pred_obj.intensity - gt.intensity))
                    total_matched_steps += 1
                    
        return {
            "mean_centroid_error": np.mean(all_centroid_errors) if all_centroid_errors else float('inf'),
            "mean_iou": np.mean(all_ious) if all_ious else 0.0,
            "mean_intensity_mae": np.mean(all_intensity_errors) if all_intensity_errors else float('inf'),
            "track_continuity": total_matched_steps / total_gt_steps if total_gt_steps > 0 else 0.0
        }

    def high_percentile_rainfall_error(self,
                                       predicted: np.ndarray,
                                       observed:  np.ndarray,
                                       percentile: float = 90.0) -> float:
        """
        MAE restricted to cells above `percentile` of observed.
        Measures tail-preservation quality. Lower = better.
        """
        threshold = np.percentile(observed, percentile)
        mask = observed > threshold
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs(predicted[mask] - observed[mask])))

    def max_value_error(self,
                        predicted: np.ndarray,
                        observed:  np.ndarray) -> float:
        """
        |max(predicted) − max(observed)| — peak preservation proxy. Lower = better.
        """
        return float(abs(float(predicted.max()) - float(observed.max())))

    def calibration_check_stub(self,
                                forecast_probabilities: np.ndarray,
                                binary_observations:   np.ndarray,
                                n_bins: int = 5) -> dict:
        """
        Stub probability calibration (Brier score + reliability table).
        Replace with proper ECE over a hindcast archive for real data.
        """
        probs  = np.clip(np.asarray(forecast_probabilities, dtype=float), 0, 1)
        obs    = np.asarray(binary_observations, dtype=float)
        brier  = float(np.mean((probs - obs) ** 2))
        bins   = np.linspace(0, 1, n_bins + 1)
        reliability = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                reliability.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0,
                                    "mean_prob": None, "mean_obs": None})
            else:
                reliability.append({
                    "bin":       f"{lo:.1f}-{hi:.1f}",
                    "n":         int(mask.sum()),
                    "mean_prob": round(float(probs[mask].mean()), 3),
                    "mean_obs":  round(float(obs[mask].mean()),   3),
                })
        return {
            "brier_score": round(brier, 4),
            "reliability": reliability,
            "note": "Stub — replace with proper ECE over hindcast archive for real data.",
        }

