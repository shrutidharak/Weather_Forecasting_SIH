import numpy as np
from typing import List, Dict
import math

try:
    from .tracker import TrackedEvent
except ImportError:
    from tracker import TrackedEvent

class EnsembleUncertaintyEstimator:
    def __init__(self):
        pass

    def estimate_uncertainty(self, ensemble_tracks: Dict[int, List[TrackedEvent]], target_event_idx: int = 0) -> dict:
        """
        ensemble_tracks: Dict mapping ensemble_member_id -> list of all tracks for that member.
        We will try to cluster/match tracks across ensembles.
        For simplicity in this prototype, we assume we just find the longest track in each member
        that overlaps with the "deterministic" member's track.
        """
        # Let's say member 0 is the control member
        if 0 not in ensemble_tracks or not ensemble_tracks[0]:
            return {"confidence": 0.0}
            
        control_tracks = ensemble_tracks[0]
        # Pick the longest track to estimate uncertainty on (for prototype demo)
        target_track = sorted(control_tracks, key=lambda x: x.duration, reverse=True)[0]
        
        matched_member_tracks = []
        
        for member_id, tracks in ensemble_tracks.items():
            if member_id == 0:
                continue
            
            # Find closest track in this member
            best_track = None
            min_dist = 10.0 # spatial threshold
            
            for track in tracks:
                # Compare final positions
                dist = math.sqrt((track.current_lat - target_track.current_lat)**2 + 
                                 (track.current_lon - target_track.current_lon)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_track = track
                    
            if best_track:
                matched_member_tracks.append(best_track)
                
        total_members = len(ensemble_tracks)
        matched_count = len(matched_member_tracks) + 1 # +1 for control
        
        probability = matched_count / total_members
        
        if not matched_member_tracks:
            return {
                "event_probability": probability,
                "location_spread_lat": 0.0,
                "location_spread_lon": 0.0,
                "intensity_spread": 0.0,
                "confidence": "Low"
            }
            
        # Collect stats at the final timestep
        lats = [target_track.current_lat] + [t.current_lat for t in matched_member_tracks]
        lons = [target_track.current_lon] + [t.current_lon for t in matched_member_tracks]
        intensities = [target_track.history[-1].intensity] + [t.history[-1].intensity for t in matched_member_tracks]
        
        lat_spread = np.std(lats)
        lon_spread = np.std(lons)
        intensity_spread = np.std(intensities)
        
        confidence = "High" if probability > 0.7 and lat_spread < 2.0 else "Medium"
        if probability < 0.4:
            confidence = "Low"
            
        return {
            "event_probability": probability,
            "location_spread_lat": float(lat_spread),
            "location_spread_lon": float(lon_spread),
            "intensity_spread": float(intensity_spread),
            "confidence": confidence,
            "consensus_centroid": (float(np.mean(lats)), float(np.mean(lons)))
        }

if __name__ == "__main__":
    print("Ensemble Uncertainty module ready.")
