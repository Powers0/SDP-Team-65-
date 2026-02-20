# SDP Team 65 — Baseball At-Bat Simulator

A pitch-by-pitch at-bat simulator built on real MLB Statcast data (2021–2024). Given a pitcher and a batter, the system predicts what pitch comes next, where it will cross the plate, and whether the batter swings — simulating a realistic at-bat sequence through a web interface.

---

## Project Structure

```
SDP-Team-65-/
├── Pitch Type Prediction/       # LSTM model: predicts next pitch type
├── Pitch Location Prediction/   # LSTM + GMM: predicts plate_x / plate_z
├── Offensive Models/            # MLP: predicts P(swing) given location + pitch type
├── api/                         # Flask backend serving all model predictions
├── csv data/                    # Raw Statcast CSVs (not committed to repo)
├── artifacts/shared/            # Shared encoder artifacts (pitcher/batter label encoders)
└── scripts/                     # Data prep utilities (serving table, player vocab)
```

---

## Setup

### Requirements
- Python 3.10+
- TensorFlow 2.x
- scikit-learn, pandas, numpy
- pybaseball (for downloading Statcast data)
- Flask, flask-cors

```bash
pip install tensorflow scikit-learn pandas numpy pybaseball flask flask-cors
```

### Downloading Data
Raw Statcast CSVs live in `csv data/`. To download 2021–2024:
```bash
python "csv data/download_full_statcast.py"
```

---

## Training Pipeline (one-time)

Run these in order. Each step produces artifacts consumed by the next.

```bash
# 1. Pitch type model
cd "Pitch Type Prediction"
python build_pitchtype_dataset.py
python train_pitchtype_model.py      # saves pitchtype_model.keras + pitch_type_probs.npy

# 2. Pitch location model
cd "../Pitch Location Prediction"
python build_pitchlocation_dataset.py
python train_pitchlocation_model.py  # saves pitch_location_model.keras
python fit_location_gmm.py           # saves location_gmm.pkl (must run after training)

# 3. Swing/take model
cd "../Offensive Models"
python build_swingtake_dataset.py
python train_swingtake_model.py

# 4. Shared artifacts
cd ..
python scripts/build_serving_table.py
python scripts/build_player_vocab.py
python scripts/build_player_names.py
```

---

## Running the App

### Start the API
```bash
cd api
python app.py
# Runs at http://localhost:5000
```

### Start the Frontend
```bash
# From the frontend directory
npm install
npm run dev
# Runs at http://localhost:5173
```

---

## Models Overview

| Model | Type | Output |
|---|---|---|
| Pitch Type | LSTM + Player Embeddings | Softmax over pitch types (FF, SL, CU, ...) |
| Pitch Location | LSTM + Gaussian NLL + GMM | plate_x, plate_z (feet from center of plate) |


---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/players` | GET | Returns list of valid pitchers and batters |
| `/api/predict` | POST | Given pitcher + batter + game context, returns next pitch prediction |

See `api/README.md` for full request/response format.

---

## Team
SDP Team 65 — Senior Design Project
