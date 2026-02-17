import numpy as np

# Serving table feature columns for previous pitch one-hot usually look like:
#   previous_pitch_FF, previous_pitch_SL, ..., previous_pitch_None
_PREV_PITCH_PREFIX = "previous_pitch_"


def _apply_user_context(serving_window, user_context: dict):
    """Overwrite the LAST row of the window with the user-selected game context
    and the last simulated pitch info (prev pitch type + prev zone), when those
    columns exist in the serving window.

    Expected user_context keys (all optional):
      balls, strikes, outs_when_up, on_1b, on_2b, on_3b, inning, score_diff,
      bat_score, fld_score,
      last_pitch_type (e.g. 'FF', 'SL', 'None'), last_zone (numeric).
    """
    if not user_context:
        return serving_window

    w = serving_window.copy()
    last_idx = w.index[-1]

    # 1) Overwrite situational/context columns
    overwrite_keys = [
        "balls",
        "strikes",
        "outs_when_up",
        "on_1b",
        "on_2b",
        "on_3b",
        "inning",
        "score_diff",
        "bat_score",
        "fld_score",
    ]

    for k in overwrite_keys:
        if k in w.columns and k in user_context and user_context[k] is not None:
            w.at[last_idx, k] = user_context[k]

    # 2) Overwrite previous pitch one-hot + previous_zone if present
    last_pitch_type = user_context.get("last_pitch_type")
    if last_pitch_type is not None:
        code = str(last_pitch_type).upper().strip()
        # Zero out all previous_pitch_* columns, then set the matching one.
        prev_cols = [c for c in w.columns if c.startswith(_PREV_PITCH_PREFIX)]
        if prev_cols:
            for c in prev_cols:
                w.at[last_idx, c] = 0.0

            # Prefer an exact match (e.g. previous_pitch_FF). Otherwise fall back to None if present.
            target = f"{_PREV_PITCH_PREFIX}{code}"
            if target in w.columns:
                w.at[last_idx, target] = 1.0
            elif f"{_PREV_PITCH_PREFIX}NONE" in w.columns:
                w.at[last_idx, f"{_PREV_PITCH_PREFIX}NONE"] = 1.0
            elif f"{_PREV_PITCH_PREFIX}None" in w.columns:
                w.at[last_idx, f"{_PREV_PITCH_PREFIX}None"] = 1.0

    if "last_zone" in user_context and user_context["last_zone"] is not None:
        if "previous_zone" in w.columns:
            w.at[last_idx, "previous_zone"] = user_context["last_zone"]

    return w

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

    serving_window = _apply_user_context(serving_window, user_context or {})

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

    # Location prediction (uses single IDs + pitch-type signal)
    # If we're sampling pitch type, feed the sampled type as a 1-hot to the location model
    # so location can vary with the chosen pitch type (instead of always conditioning on the
    # full distribution).
    if sample_pitch_type:
        pt_for_loc = np.zeros_like(pt_probs, dtype=np.float32)
        pt_for_loc[pt_idx] = 1.0
    else:
        pt_for_loc = pt_probs

    loc_pred_scaled = location_model.predict(
        [Xloc, p_ids[:, 0], b_ids[:, 0], pt_for_loc[np.newaxis, :]],
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
        "pitch_type_prob": float(pt_probs[pt_idx]),
        "pitch_type_probs": {str(artifacts["pt_label_enc"].classes_[i]): float(pt_probs[i]) for i in range(len(pt_probs))},
        "location": {"plate_x": float(loc_pred[0]), "plate_z": float(loc_pred[1])},
        "context": context
    }