# Extreme Weather Alert API — curl Examples
## SIH 26078 Prototype v2.0

Start the server first:
```bash
uvicorn api.main:app --reload --port 8000
```

---

## Health Check

```bash
curl http://localhost:8000/health
```
**Response:**
```json
{"status": "ok", "version": "2.0.0", "mode": "synthetic-data-prototype"}
```

---

## Trigger a Forecast Run

```bash
curl -X POST http://localhost:8000/forecast/run \
  -H "Content-Type: application/json" \
  -d '{"scenario": "cyclone", "ensemble_members": 10}'
```
**Response:**
```json
{"run_id": "abc123...", "scenario": "cyclone", "status": "queued", "started_at": 1727000000.0}
```

---

## Check Run Status

```bash
curl http://localhost:8000/forecast/run/abc123...
```
**Response:**
```json
{"run_id": "abc123...", "status": "done", "summary": {"n_events": 3, ...}}
```

---

## List All Detected Events

```bash
curl http://localhost:8000/events
```

Filter by scenario:
```bash
curl "http://localhost:8000/events?scenario=cyclone&limit=10"
```

---

## Get a Single Event (Full 4D Schema)

```bash
curl http://localhost:8000/events/{event_id}
```
**Response:**
```json
{
  "event_id": "...",
  "event_type": "cyclone",
  "forecast_time": 24,
  "latitude_range": [14.0, 22.0],
  "longitude_range": [16.0, 26.0],
  "vertical_range": [500.0, 1013.0],
  "t_start": 2,
  "t_end": 38,
  "severity": 0.81,
  "confidence": 0.73,
  "probability_exceedance": 0.73,
  "physics_plausibility": 0.87,
  "risk_level": "severe"
}
```

---

## Get Event Track (Spatial History)

```bash
curl http://localhost:8000/events/{event_id}/track
```
**Response:**
```json
{
  "event_id": "...",
  "event_type": "cyclone",
  "t_start": 2,
  "t_end": 38,
  "latitude_range": [14.0, 22.0],
  "longitude_range": [16.0, 26.0],
  "vertical_range": [500.0, 1013.0],
  "centroid_lat": 18.5,
  "centroid_lon": 21.0,
  "speed": 0.52,
  "direction_deg": -135.0
}
```

---

## Get Impact Grid (2D spatial impact map)

```bash
curl "http://localhost:8000/events/{event_id}/impact-grid?grid_h=32&grid_w=32"
```
**Response:**
```json
{
  "event_id": "...",
  "grid_h": 32,
  "grid_w": 32,
  "impact": [[0.01, 0.03, ...], ...],
  "units": "normalised severity×plausibility [0,1]"
}
```

---

## Get Alerts for an Event

```bash
curl http://localhost:8000/events/{event_id}/alerts
```
**Response:**
```json
{
  "event_id": "...",
  "event_type": "cyclone",
  "risk_level": "severe",
  "colour": "#FF5722",
  "recommended_action": "Warning issued — activate district-level response.",
  "severity": 0.81,
  "confidence": 0.73,
  "physics_plausibility": 0.87,
  "probability_exceedance": 0.73
}
```

---

## Risk Level Mapping

| risk_level | Colour    | Trigger condition                             |
|------------|-----------|-----------------------------------------------|
| `low`      | 🟢 Green  | severity × confidence < 0.20                  |
| `moderate` | 🟡 Amber  | 0.20–0.50 **or** high severity + low confidence |
| `severe`   | 🟠 Orange | 0.50–0.75                                     |
| `extreme`  | 🔴 Red    | > 0.75 (requires both high sev AND conf)      |

> **Note:** Low-confidence (< 0.3) severe events are capped at `moderate`
> to prevent false high-priority alerts — matching the SIH brief's guidance.
