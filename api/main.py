"""
api/main.py — Extreme Weather Alert API
=========================================
FastAPI application exposing the ST-GNN pipeline as a REST service.

Endpoints:
    GET  /health                         — liveness check
    GET  /events                         — list all cached detected events
    GET  /events/{event_id}              — single event detail (full schema)
    GET  /events/{event_id}/track        — full detection history
    GET  /events/{event_id}/impact-grid  — 2-D impact grid (H×W JSON)
    GET  /events/{event_id}/alerts       — risk-level alert dict
    POST /forecast/run                   — trigger a pipeline run + cache results

Storage: in-memory SQLite (no Postgres/PostGIS needed at prototype stage).

Run:
    uvicorn api.main:app --reload --port 8000
"""
import sys
import os
import json
import sqlite3
import time
import uuid
from typing import Optional, List, Dict, Any

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# ── Pipeline import ────────────────────────────────────────────────────────────
from run_pipeline import run_pipeline

# ── App + CORS ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Extreme Weather Alert API — SIH 26078",
    description=(
        "Prototype REST interface over the Spatio-Temporal GNN pipeline. "
        "All data is synthetic; see README §5 for real-data swap points."
    ),
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SQLite in-memory cache ─────────────────────────────────────────────────────
_DB_PATH = ":memory:"   # swap to a file path for persistence between restarts


def _get_db() -> sqlite3.Connection:
    """Return (or create) the module-level in-memory SQLite connection."""
    if not hasattr(_get_db, "_conn") or _get_db._conn is None:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id    TEXT PRIMARY KEY,
                scenario    TEXT,
                run_id      TEXT,
                created_at  REAL,
                payload     TEXT    -- JSON blob of the full event dict
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id      TEXT PRIMARY KEY,
                scenario    TEXT,
                started_at  REAL,
                finished_at REAL,
                status      TEXT,
                summary     TEXT    -- JSON
            )
        """)
        conn.commit()
        _get_db._conn = conn
    return _get_db._conn

_get_db._conn = None   # initialise


# ── Pydantic request/response models ──────────────────────────────────────────

class ForecastRunRequest(BaseModel):
    scenario: str = "B"        # A / B / cyclone / heatwave
    ensemble_members: int = 10


class RunStatus(BaseModel):
    run_id:     str
    scenario:   str
    status:     str
    n_events:   int = 0
    started_at: float
    finished_at: Optional[float] = None


# ── Background pipeline task ───────────────────────────────────────────────────

def _pipeline_task(run_id: str, scenario: str):
    """Executes pipeline in the background, stores results in SQLite."""
    db = _get_db()
    db.execute(
        "UPDATE runs SET status=? WHERE run_id=?", ("running", run_id)
    )
    db.commit()
    try:
        result = run_pipeline(scenario=scenario, return_results=True)
        events: List[Dict] = result.get("events", [])
        for ev in events:
            ev_id = ev.get("event_id", str(uuid.uuid4()))
            db.execute(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?)",
                (ev_id, scenario, run_id, time.time(), json.dumps(ev))
            )
        summary = {
            "n_events":       len(events),
            "scenario":       scenario,
            "metrics":        result.get("metrics", {}),
            "uncertainty":    result.get("uncertainty", {}),
        }
        db.execute(
            "UPDATE runs SET status=?, finished_at=?, summary=? WHERE run_id=?",
            ("done", time.time(), json.dumps(summary), run_id)
        )
        db.commit()
    except Exception as exc:
        db.execute(
            "UPDATE runs SET status=?, finished_at=? WHERE run_id=?",
            (f"error: {exc}", time.time(), run_id)
        )
        db.commit()
        raise


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    """Liveness check — always returns 200 if the server is up."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "mode": "synthetic-data-prototype",
        "timestamp": time.time(),
    }


@app.post("/forecast/run", tags=["pipeline"], response_model=RunStatus)
def trigger_forecast(req: ForecastRunRequest, background_tasks: BackgroundTasks):
    """
    Trigger a full pipeline run for the requested scenario.
    Results are cached in SQLite and queryable via /events.
    Returns immediately with a run_id to poll status.
    """
    run_id = str(uuid.uuid4())
    db = _get_db()
    db.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?)",
        (run_id, req.scenario, time.time(), None, "queued", None)
    )
    db.commit()
    background_tasks.add_task(_pipeline_task, run_id, req.scenario)
    return RunStatus(
        run_id=run_id,
        scenario=req.scenario,
        status="queued",
        started_at=time.time(),
    )


@app.get("/forecast/run/{run_id}", tags=["pipeline"])
def get_run_status(run_id: str):
    """Poll the status of a pipeline run."""
    db  = _get_db()
    row = db.execute(
        "SELECT * FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if not row:
        raise HTTPException(404, detail=f"Run '{run_id}' not found.")
    d = dict(row)
    d["summary"] = json.loads(d["summary"]) if d["summary"] else {}
    return d


@app.get("/events", tags=["events"])
def list_events(scenario: Optional[str] = None, limit: int = 50):
    """
    List all cached detected events.
    Optionally filter by scenario (A / B / cyclone / heatwave).
    """
    db = _get_db()
    if scenario:
        rows = db.execute(
            "SELECT payload FROM events WHERE scenario=? LIMIT ?",
            (scenario, limit)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT payload FROM events LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


@app.get("/events/{event_id}", tags=["events"])
def get_event(event_id: str):
    """Return the full 4D schema for a single event."""
    db  = _get_db()
    row = db.execute(
        "SELECT payload FROM events WHERE event_id=?", (event_id,)
    ).fetchone()
    if not row:
        raise HTTPException(404, detail=f"Event '{event_id}' not found.")
    return json.loads(row["payload"])


@app.get("/events/{event_id}/track", tags=["events"])
def get_event_track(event_id: str):
    """
    Return the full detection history (centroid positions over time)
    for a tracked event.
    """
    ev = get_event(event_id)
    # For ST-GNN events, history is encoded in t_start/t_end + bboxes.
    return {
        "event_id":       event_id,
        "event_type":     ev.get("event_type"),
        "t_start":        ev.get("t_start", ev.get("start_time", 0)),
        "t_end":          ev.get("t_end",   ev.get("end_time",   0)),
        "latitude_range": ev.get("latitude_range"),
        "longitude_range":ev.get("longitude_range"),
        "vertical_range": ev.get("vertical_range"),
        "centroid_lat":   ev.get("centroid_lat"),
        "centroid_lon":   ev.get("centroid_lon"),
        "speed":          ev.get("speed"),
        "direction_deg":  ev.get("direction_deg"),
    }


@app.get("/events/{event_id}/impact-grid", tags=["events"])
def get_impact_grid(event_id: str, grid_h: int = 32, grid_w: int = 32):
    """
    Return a synthetic 2-D impact grid (severity × plausibility) for the event.
    Values are in [0, 1]. In a real deployment this would be interpolated from
    the high-resolution downscaled output.
    """
    ev = get_event(event_id)
    lat_r  = ev.get("latitude_range",  [0, grid_h])
    lon_r  = ev.get("longitude_range", [0, grid_w])
    sev    = ev.get("severity",  0.5)
    phys   = ev.get("physics_plausibility", 1.0)

    # Build a simple Gaussian impact blob over the bbox
    impact = np.zeros((grid_h, grid_w))
    c_lat  = (lat_r[0] + lat_r[1]) / 2
    c_lon  = (lon_r[0] + lon_r[1]) / 2
    for i in range(grid_h):
        for j in range(grid_w):
            d2 = (i - c_lat)**2 + (j - c_lon)**2
            impact[i, j] = sev * phys * np.exp(-d2 / 18.0)

    return {
        "event_id": event_id,
        "grid_h":   grid_h,
        "grid_w":   grid_w,
        "impact":   impact.round(4).tolist(),
        "units":    "normalised severity×plausibility [0,1]",
    }


@app.get("/events/{event_id}/alerts", tags=["events"])
def get_event_alerts(event_id: str):
    """
    Return the risk-level alert dict for a single event.
    Matches the severity × confidence rule from §5.
    """
    ev         = get_event(event_id)
    risk       = ev.get("risk_level", "unknown")
    sev        = ev.get("severity",  0.0)
    conf       = ev.get("confidence", 0.0)
    event_type = ev.get("event_type", "unknown")

    colour_map = {
        "low":      "#4CAF50",
        "moderate": "#FFC107",
        "severe":   "#FF5722",
        "extreme":  "#B71C1C",
    }
    action_map = {
        "low":      "Monitor — no immediate action required.",
        "moderate": "Advisory issued — prepare response teams.",
        "severe":   "Warning issued — activate district-level response.",
        "extreme":  "EMERGENCY ALERT — activate full disaster response protocol.",
    }
    return {
        "event_id":         event_id,
        "event_type":       event_type,
        "risk_level":       risk,
        "colour":           colour_map.get(risk, "#9E9E9E"),
        "recommended_action": action_map.get(risk, "Unknown risk level."),
        "severity":         round(sev, 4),
        "confidence":       round(conf, 4),
        "physics_plausibility": ev.get("physics_plausibility", 1.0),
        "probability_exceedance": ev.get("probability_exceedance", 0.0),
    }
