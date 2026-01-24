import numpy as np

def ensure_columns(df_sub, required_cols):
    for col in required_cols:
        if col not in df_sub.columns:
            df_sub[col] = 0.0
    return df_sub[required_cols]

def predict_next(serving_window,
                 artifacts,
                 seq_len: int):
    pt_features  = artifacts["pt_features"]
    pt_scaler_X  = artifacts["pt_scaler_X"]
    pt_label_enc = artifacts["pt_label_enc"]

    loc_features = artifacts["loc_features"]
    loc_scaler_X = artifacts["loc_scaler_X"]
    loc_scaler_Y = artifacts["loc_scaler_Y"]

    pitchtype_model = artifacts["pitchtype_model"]
    location_model  = artifacts["location_model"]

    # Build feature matrices aligned to training features
    sub_pt  = ensure_columns(serving_window.copy(), pt_features)
    sub_loc = ensure_columns(serving_window.copy(), loc_features)

    Xpt  = pt_scaler_X.transform(sub_pt.values)[np.newaxis, :, :]
    Xloc = loc_scaler_X.transform(sub_loc.values)[np.newaxis, :, :]

    # Embedding IDs from serving table (already stable from shared encoders)
    p_idx = int(serving_window["pitcher_embed"].iloc[-1])
    b_idx = int(serving_window["batter_embed"].iloc[-1])

    p_ids = np.full((1, seq_len), p_idx, dtype=np.int32)
    b_ids = np.full((1, seq_len), b_idx, dtype=np.int32)

    # Pitch type prediction
    pt_probs = pitchtype_model.predict([Xpt, p_ids, b_ids], verbose=0)[0]
    pt_pred = pt_label_enc.classes_[int(np.argmax(pt_probs))]

    # Location prediction (uses single IDs + pitch-type probs)
    loc_pred_scaled = location_model.predict(
        [Xloc, p_ids[:, 0], b_ids[:, 0], pt_probs[np.newaxis, :]],
        verbose=0
    )
    loc_pred = loc_scaler_Y.inverse_transform(loc_pred_scaled)[0]

    # Context for UI (last pitch state is the “current” state)
    last = serving_window.iloc[-1]
    context = {}
    for k in ["game_date","inning","balls","strikes","outs_when_up","on_1b","on_2b","on_3b","score_diff"]:
        if k in last.index:
            context[k] = int(last[k]) if str(last[k]).isdigit() else last[k]

    return {
        "pitch_type": str(pt_pred),
        "pitch_type_probs": {str(artifacts["pt_label_enc"].classes_[i]): float(pt_probs[i]) for i in range(len(pt_probs))},
        "location": {"plate_x": float(loc_pred[0]), "plate_z": float(loc_pred[1])},
        "context": context
    }