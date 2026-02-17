import numpy as np

def ensure_columns(df_sub, required_cols):
    for col in required_cols:
        if col not in df_sub.columns:
            df_sub[col] = 0.0
    return df_sub[required_cols]

def predict_next(
    serving_window,
    artifacts,
    seq_len: int,
    pitcher_mlbam: int,
    batter_mlbam: int,
    user_context: dict | None = None,
    sample_pitch_type: bool = False,
):
    pt_features  = artifacts["pt_features"]
    pt_scaler_X  = artifacts["pt_scaler_X"]
    pt_label_enc = artifacts["pt_label_enc"]

    loc_features = artifacts["loc_features"]
    loc_scaler_X = artifacts["loc_scaler_X"]
    loc_scaler_Y = artifacts["loc_scaler_Y"]

    pitchtype_model = artifacts["pitchtype_model"]
    location_model  = artifacts["location_model"]

    # Apply user-selected context (balls/strikes/outs/bases/inning/score) to the LAST row
    # so the model conditions on the situation the user picked, not just whatever the
    # serving table's latest pitch happened to be.
    if user_context:
        # Work on a copy so we don't mutate the global SERVING_DF window
        serving_window = serving_window.copy()

        # Only overwrite columns that exist in the serving table.
        # (Keeps this robust across different feature sets.)
        overwrite_keys = [
            "balls",
            "strikes",
            "outs_when_up",
            "on_1b",
            "on_2b",
            "on_3b",
            "inning",
            "score_diff",
        ]

        last_idx = serving_window.index[-1]
        for k in overwrite_keys:
            if k in serving_window.columns and k in user_context and user_context[k] is not None:
                serving_window.at[last_idx, k] = user_context[k]

    # Build feature matrices aligned to training features
    sub_pt  = ensure_columns(serving_window.copy(), pt_features)
    sub_loc = ensure_columns(serving_window.copy(), loc_features)

    Xpt  = pt_scaler_X.transform(sub_pt.values)[np.newaxis, :, :]
    Xloc = loc_scaler_X.transform(sub_loc.values)[np.newaxis, :, :]

    # Embedding IDs from encoders (ALWAYS use selected players)
    p_le = artifacts["pitcher_le"]
    b_le = artifacts["batter_le"]

    try:
        p_idx = int(p_le.transform([pitcher_mlbam])[0])
    except Exception:
        p_idx = 0  # fallback index

    try:
        b_idx = int(b_le.transform([batter_mlbam])[0])
    except Exception:
        b_idx = 0  # fallback index

    p_ids = np.full((1, seq_len), p_idx, dtype=np.int32)
    b_ids = np.full((1, seq_len), b_idx, dtype=np.int32)

    # Pitch type prediction
    pt_probs = pitchtype_model.predict([Xpt, p_ids, b_ids], verbose=0)[0]
    if sample_pitch_type:
        # Stochastic draw (makes repeated clicks less deterministic)
        pt_idx = int(np.random.choice(len(pt_probs), p=pt_probs / np.sum(pt_probs)))
    else:
        # Deterministic (argmax)
        pt_idx = int(np.argmax(pt_probs))

    pt_pred = pt_label_enc.classes_[pt_idx]

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
            v = last[k]

            # convert numpy scalars to python scalars (jsonify-safe)
            if isinstance(v, (np.integer, np.floating)):
                v = v.item()

            context[k] = v

    return {
        "pitch_type": str(pt_pred),
        "pitch_type_idx": int(pt_idx),
        "pitch_type_probs": {str(artifacts["pt_label_enc"].classes_[i]): float(pt_probs[i]) for i in range(len(pt_probs))},
        "location": {"plate_x": float(loc_pred[0]), "plate_z": float(loc_pred[1])},
        "context": context
    }