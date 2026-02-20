# API

Flask backend that serves all model predictions to the frontend. Loads all trained model artifacts at startup and exposes two endpoints.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask app, endpoint definitions, repertoire masking logic |
| `src/config.py` | All file paths and constants (seq length, artifact dirs) |
| `src/artifacts.py` | Loads all model artifacts into a single dict at startup |
| `src/inference.py` | Core prediction logic: pitch type + location + GMM sampling |
| `src/serving_data.py` | Loads the serving table, builds per-pitcher/batter windows |

---

## Running

```bash
cd api
python app.py
# Runs at http://localhost:5000
```

The frontend expects the API at `http://localhost:5000/api/*`.

---

## Endpoints

### `GET /api/players`

Returns the list of pitchers and batters available in the simulator.

Players are filtered by minimum activity thresholds:
- Pitchers: must have thrown at least **200 pitches** in the dataset
- Batters: must have at least **50 plate appearances** in the dataset
- Real pitchers (>= 200 pitches) are excluded from the batter list (except Ohtani)

**Response:**
```json
{
  "pitchers": [
    { "id": 594798, "label": "Jacob deGrom", "throws": "R" }
  ],
  "batters": [
    { "id": 596019, "label": "Javier Baez", "bats": "R", "sz_top": 3.37, "sz_bot": 1.55 }
  ]
}
```

---

### `POST /api/predict`

Given a pitcher, batter, and optional game context, returns the next pitch prediction.

**Request body:**
```json
{
  "pitcher_mlbam": 594798,
  "batter_mlbam": 596019,
  "user_context": {
    "balls": 1,
    "strikes": 2,
    "outs_when_up": 1,
    "on_1b": 0,
    "on_2b": 0,
    "on_3b": 0,
    "inning": 5,
    "score_diff": -1,
    "last_pitch_type": "FF",
    "last_zone": 6
  }
}
```

All fields in `user_context` are optional. If omitted, the model uses the most recent real game state for that pitcher/batter pair from the serving table.

**Response:**
```json
{
  "pitch_type": "SL",
  "pitch_type_prob": 0.41,
  "pitch_type_probs": { "FF": 0.28, "SL": 0.41, "CU": 0.12, ... },
  "location": { "plate_x": 0.62, "plate_z": 1.83 },
  "location_mean": { "plate_x": 0.54, "plate_z": 1.91 },
  "location_std": { "plate_x": 0.21, "plate_z": 0.29 },
  "context": {
    "balls": 1, "strikes": 2, "outs_when_up": 1,
    "inning": 5, "score_diff": -1
  },
  "context_label": "pitcher_batter"
}
```

| Field | Description |
|---|---|
| `pitch_type` | Predicted pitch type code (e.g. `"FF"`, `"SL"`) |
| `pitch_type_prob` | Probability of the predicted type after repertoire masking |
| `pitch_type_probs` | Full probability distribution over all pitch types |
| `location.plate_x` | Horizontal position in feet (negative = catcher's left / arm side) |
| `location.plate_z` | Vertical position in feet above the ground |
| `location_mean` | LSTM model's raw predicted mean (before GMM sampling) |
| `location_std` | Model's predicted uncertainty (std dev) in each dimension |
| `context_label` | Indicates which fallback was used: `"pitcher_batter"`, `"pitcher_only"`, or `"league_avg"` |

---

## Coordinate System

`plate_x` and `plate_z` follow Statcast conventions:
- `plate_x = 0` is the center of the plate
- `plate_x < 0` is toward the catcher's left (pitcher's arm side for a RHP)
- `plate_z = 0` is ground level; typical strike zone is roughly `1.5–3.5 ft`
- Home plate is **17 inches wide**, so `±0.708 ft` covers the strike zone horizontally

---

## Repertoire Masking

After the model produces raw pitch type probabilities, the API applies a **repertoire mask**: any pitch type the pitcher has never thrown in the dataset is zeroed out, and the remaining probabilities are renormalized. This prevents the model from predicting a pitch the pitcher doesn't throw (e.g., a knuckleball for deGrom).

Pitch type aliases are normalized at both masking and training time (`FO → FS`, `SF → FS`) to ensure consistency.

---

## Inference Pipeline (per request)

1. Look up the pitcher/batter's recent pitch window from the serving table
2. Overwrite the last row with `user_context` (count, base state, etc.)
3. Run the **pitch type model** → get softmax probability vector
4. Apply repertoire mask → renormalize → stochastically sample pitch type
5. Run the **location model** → get `[mu_x, mu_z, log_var_x, log_var_z]`
6. Apply targeting bias heuristic (count-based nudges to correct mean regression)
7. Sample final `(plate_x, plate_z)` from the **GMM** using `component_temperature=0.35`
8. Return prediction + full probability breakdown to frontend

---

## Configuration (`src/config.py`)

| Constant | Value | Description |
|---|---|---|
| `SEQ_LEN` | 5 | Number of pitches of context fed to both models |
| `PT_DIR` | `Pitch Type Prediction/artifacts/` | Pitch type model artifacts |
| `LOC_DIR` | `Pitch Location Prediction/artifacts/` | Location model artifacts |
| `SHARED_DIR` | `artifacts/shared/` | Pitcher/batter label encoders, player names |
| `LOCATION_GMM_PATH` | `Pitch Location Prediction/artifacts/location_gmm.pkl` | GMM artifact |
