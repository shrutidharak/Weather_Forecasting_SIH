import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data.synthetic_generator import (
    SyntheticDataLoader, SyntheticWeatherGenerator,
    N_CHANNELS, EVENT_TYPES
)


def test_scenario_a_dimensions_and_features():
    loader = SyntheticDataLoader(scenario="A", seed=0)
    low_res, high_res, gt = loader.load_data()

    # 6-channel tensors
    assert low_res.shape  == (10, 40, 32, 32, 6), f"Unexpected low_res shape: {low_res.shape}"
    assert high_res.shape == (10, 40, 64, 64, 6), f"Unexpected high_res shape: {high_res.shape}"
    assert low_res.shape[-1]  == N_CHANNELS == 6
    assert high_res.shape[-1] == N_CHANNELS == 6

    # Ground truth: one record per time step (ensemble 0)
    assert len(gt) == 40, f"Expected 40 GT records, got {len(gt)}"


def test_nan_and_infinite_values():
    loader = SyntheticDataLoader(scenario="A", seed=1)
    low_res, high_res, _ = loader.load_data()

    assert not np.isnan(low_res).any(),  "Found NaN in low_res"
    assert not np.isnan(high_res).any(), "Found NaN in high_res"
    assert not np.isinf(low_res).any(),  "Found Inf in low_res"
    assert not np.isinf(high_res).any(), "Found Inf in high_res"


def test_ensemble_diversity():
    loader = SyntheticDataLoader(scenario="A", seed=2)
    low_res, _, _ = loader.load_data()

    diff = np.abs(low_res[0] - low_res[1]).sum()
    assert diff > 0, "Ensemble members are perfectly identical"


def test_temporal_evolution_and_movement():
    loader = SyntheticDataLoader(scenario="A", seed=3)
    _, _, gt = loader.load_data()

    t0_lat, t0_lon = gt[0].centroid_lat, gt[0].centroid_lon
    t1_lat, t1_lon = gt[-1].centroid_lat, gt[-1].centroid_lon
    assert t0_lat != t1_lat or t0_lon != t1_lon, "Event did not move over time"


def test_valid_bounding_boxes():
    loader = SyntheticDataLoader(scenario="A", seed=4)
    _, _, gt = loader.load_data()

    for record in gt:
        min_lat, min_lon, max_lat, max_lon = record.bbox
        assert min_lat <= max_lat, f"Invalid bbox lat: {record.bbox}"
        assert min_lon <= max_lon, f"Invalid bbox lon: {record.bbox}"


def test_terrain_shape_and_values():
    loader = SyntheticDataLoader(scenario="A", seed=5)
    terrain = loader.get_terrain()

    assert terrain.shape == (32, 32, 2), f"Unexpected terrain shape: {terrain.shape}"
    # Elevation channel: 0–3000 m
    elev = terrain[:, :, 0]
    assert elev.min() >= 0.0,    f"Negative elevation: {elev.min()}"
    assert elev.max() <= 3001.0, f"Elevation too high: {elev.max()}"
    # Coastline distance: >= 0
    coast = terrain[:, :, 1]
    assert coast.min() >= 0.0, "Negative coastline distance"
    # No NaN/Inf
    assert not np.isnan(terrain).any()
    assert not np.isinf(terrain).any()


def test_cyclone_scenario():
    loader = SyntheticDataLoader(scenario="cyclone", seed=6)
    lr, hr, gt = loader.load_data()

    assert lr.shape  == (10, 40, 32, 32, 6)
    assert hr.shape  == (10, 40, 64, 64, 6)
    assert all(g.event_type == "cyclone" for g in gt), "All GT should be cyclone"
    # Cyclone: pressure channel should have negative anomaly near centroid
    t0 = gt[0]
    ci = int(round(t0.centroid_lat))
    cj = int(round(t0.centroid_lon))
    ci = np.clip(ci, 0, 31); cj = np.clip(cj, 0, 31)
    pres_center = lr[0, 0, ci, cj, 4]   # CH_PRES
    assert pres_center < 0, f"Expected negative pressure anomaly, got {pres_center:.3f}"


def test_heatwave_scenario():
    loader = SyntheticDataLoader(scenario="heatwave", seed=7)
    lr, hr, gt = loader.load_data()

    assert lr.shape == (10, 40, 32, 32, 6)
    assert all(g.event_type == "heatwave" for g in gt)
    # Heatwave: temperature channel should be elevated near centroid
    t0 = gt[0]
    ci = int(round(t0.centroid_lat))
    cj = int(round(t0.centroid_lon))
    ci = np.clip(ci, 0, 31); cj = np.clip(cj, 0, 31)
    temp_center = lr[0, 0, ci, cj, 3]   # CH_TEMP
    assert temp_center > 0, f"Expected positive temp anomaly, got {temp_center:.3f}"


def test_derived_fields():
    gen = SyntheticWeatherGenerator(seed=8)
    lr, _, _ = gen.generate_scenario_a()
    frame = lr[0, 0]   # (H, W, 6)

    ws = SyntheticWeatherGenerator.compute_wind_speed(frame)
    mc = SyntheticWeatherGenerator.compute_moisture_convergence(frame)

    assert ws.shape == (32, 32), f"Wind speed shape: {ws.shape}"
    assert mc.shape == (32, 32), f"Moisture convergence shape: {mc.shape}"
    assert not np.isnan(ws).any()
    assert not np.isnan(mc).any()
    assert (ws >= 0).all(), "Wind speed should be non-negative"


def test_event_type_in_ground_truth():
    for scen, expected_type in [("A", "extreme_rainfall"), ("cyclone", "cyclone"),
                                 ("heatwave", "heatwave")]:
        loader = SyntheticDataLoader(scenario=scen, seed=9)
        _, _, gt = loader.load_data()
        types = {g.event_type for g in gt}
        assert expected_type in types, f"Scenario {scen}: expected {expected_type}, got {types}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
