import pickle
from tensorflow.keras.models import load_model
import json
import os

def load_all(PT_DIR, LOC_DIR, SHARED_DIR):
    pitchtype_model = load_model(PT_DIR + "pitchtype_model.keras")
    location_model  = load_model(LOC_DIR + "pitch_location_model.keras")

    pt_features  = pickle.load(open(PT_DIR + "features.pkl", "rb"))
    pt_scaler_X  = pickle.load(open(PT_DIR + "scaler.pkl", "rb"))
    pt_label_enc = pickle.load(open(PT_DIR + "label_encoder.pkl", "rb"))

    loc_features = pickle.load(open(LOC_DIR + "features.pkl", "rb"))
    loc_scaler_X = pickle.load(open(LOC_DIR + "scaler_X.pkl", "rb"))
    loc_scaler_Y = pickle.load(open(LOC_DIR + "scaler_Y.pkl", "rb"))

    pitcher_le = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
    batter_le  = pickle.load(open(SHARED_DIR + "batter_le.pkl", "rb"))

    names_path = os.path.join(SHARED_DIR, "player_names.json")
    with open(names_path, "r") as f:
        player_names = json.load(f)

    return {
        "pitchtype_model": pitchtype_model,
        "location_model": location_model,
        "pt_features": pt_features,
        "pt_scaler_X": pt_scaler_X,
        "pt_label_enc": pt_label_enc,
        "loc_features": loc_features,
        "loc_scaler_X": loc_scaler_X,
        "loc_scaler_Y": loc_scaler_Y,
        "pitcher_le": pitcher_le,
        "batter_le": batter_le,
        "player_names": player_names,
    }