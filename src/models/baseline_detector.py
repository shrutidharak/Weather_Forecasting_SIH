import numpy as np
from scipy.ndimage import label, find_objects, center_of_mass
from typing import List, Dict, Tuple
from dataclasses import dataclass
import uuid

@dataclass
class DetectedObject:
    object_id: str
    timestamp: int
    centroid_lat: float
    centroid_lon: float
    bbox: Tuple[float, float, float, float]  # min_lat, min_lon, max_lat, max_lon
    area: int
    intensity: float

class BaselineDetector:
    def __init__(self, climatology, feature_idx: int = 0, percentile: int = 99):
        """
        Baseline rule-based detector.
        feature_idx: Which feature to detect extremes on (e.g., 0 for precipitation)
        percentile: The threshold percentile to use for the binary mask
        """
        self.climatology = climatology
        self.feature_idx = feature_idx
        self.percentile = percentile

    def detect(self, forecast: np.ndarray, time_step: int = 0) -> List[DetectedObject]:
        """
        forecast shape: (lat, lon, features) 
        This is for a single time step and a single ensemble member.
        """
        # Step 1: Anomaly calculation & Thresholding (Binary Mask)
        # forecast > threshold
        if self.percentile == 90:
            threshold = self.climatology.percentile_90[:, :, self.feature_idx]
        elif self.percentile == 95:
            threshold = self.climatology.percentile_95[:, :, self.feature_idx]
        elif self.percentile == 99:
            threshold = self.climatology.percentile_99[:, :, self.feature_idx]
        else:
            raise ValueError("Unsupported percentile.")
            
        feature_data = forecast[:, :, self.feature_idx]
        binary_mask = feature_data > threshold
        
        # Step 2: Connected Components
        labeled_array, num_features = label(binary_mask)
        
        # Step 3: Object extraction
        detected_objects = []
        if num_features == 0:
            return detected_objects
            
        objects_slices = find_objects(labeled_array)
        centroids = center_of_mass(feature_data, labeled_array, range(1, num_features + 1))
        
        # If center_of_mass returns a single tuple instead of a list of tuples
        if isinstance(centroids, tuple):
            centroids = [centroids]
            
        for i, slice_tuple in enumerate(objects_slices):
            lat_slice, lon_slice = slice_tuple
            
            # Area in pixels
            component_mask = (labeled_array == (i + 1))
            area = int(np.sum(component_mask))
            
            # Max intensity within the object
            intensity = float(np.max(feature_data[component_mask]))
            
            # Centroid
            centroid_lat, centroid_lon = centroids[i]
            
            # Bounding box
            min_lat, max_lat = lat_slice.start, lat_slice.stop
            min_lon, max_lon = lon_slice.start, lon_slice.stop
            
            obj = DetectedObject(
                object_id=str(uuid.uuid4()),
                timestamp=time_step,
                centroid_lat=float(centroid_lat),
                centroid_lon=float(centroid_lon),
                bbox=(float(min_lat), float(min_lon), float(max_lat), float(max_lon)),
                area=area,
                intensity=intensity
            )
            detected_objects.append(obj)
            
        return detected_objects

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    from data.synthetic_generator import SyntheticDataLoader
    from data.climatology import SyntheticClimatology
    
    loader = SyntheticDataLoader(scenario="A")
    low_res, _, _ = loader.load_data()
    
    climatology = SyntheticClimatology()
    climatology.generate_historical_baseline()
    
    detector = BaselineDetector(climatology, feature_idx=0, percentile=99)
    
    # Test on Ensemble 0, Time 10
    time_step = 10
    forecast_frame = low_res[0, time_step]
    
    objects = detector.detect(forecast_frame, time_step=time_step)
    print(f"Detected {len(objects)} objects at time {time_step}.")
    if objects:
        print(f"First object: {objects[0]}")
