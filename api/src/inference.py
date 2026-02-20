import numpy as np

# Serving table feature columns for previous pitch one-hot usually look like:
#   previous_pitch_FF, previous_pitch_SL, ..., previous_pitch_None

_PREV_PITCH_PREFIX = "previous_pitch_"

# Simple physical-ish bounds so noise doesn't go insane.
PLATE_X_BOUNDS = (-2.0, 2.0)
PLATE_Z_BOUNDS = (0.5, 4.5)

# ---------------------------------------------------------------------------
# GMM-based location sampler
# ---------------------------------------------------------------------------

def _nearest_component(gmm, mean_x: float, mean_z: float) -> int:
    """Return the index of the GMM component whose mean is closest to the
    model-predicted (mean_x, mean_z).  This preserves the model's directional
    signal while using the GMM for realistic spread."""
    query = np.array([[mean_x, mean_z]])
    dists = np.sum((gmm.means_ - query) ** 2, axis=1)
    return int(np.argmin(dists))


def sample_location_gmm(
    mean_x: float,
    mean_z: float,
    pitch_type: str,
    gmm_dict: dict,
    rng: np.random.Generator,
    # How strongly to weight model mean vs. pure component sampling.
    # 0.0 = always pick nearest component (full model guidance)
    # 1.0 = sample component by mixture weights (ignore model mean)
    component_temperature: float = 0.0,
) -> tuple[float, float]:
    """Sample a pitch location using the GMM fitted on real Statcast data.
    If no GMM is available for the pitch type, falls back to a simple
    Gaussian sample around the model mean with realistic std devs.
    """
    code = str(pitch_type).upper().strip()
    gmm = gmm_dict.get(code) if gmm_dict else None

    if gmm is None:
        # Fallback: sample with realistic spread (wider than before)
        FALLBACK_STD = {"x": 0.70, "z": 0.75}
        x = float(np.clip(rng.normal(mean_x, FALLBACK_STD["x"]), *PLATE_X_BOUNDS))
        z = float(np.clip(rng.normal(mean_z, FALLBACK_STD["z"]), *PLATE_Z_BOUNDS))
        return x, z

    n = gmm.n_components

    if component_temperature <= 0.0:
        # Pure model-guidance: always pick the nearest component
        comp_idx = _nearest_component(gmm, mean_x, mean_z)
    else:
        # Blend: weighted random between nearest-component and mixture weights
        # component_temperature=1 → pure mixture weights (ignores model mean)
        nearest = _nearest_component(gmm, mean_x, mean_z)
        one_hot = np.zeros(n)
        one_hot[nearest] = 1.0
        blend = (1.0 - component_temperature) * one_hot + component_temperature * gmm.weights_
        blend = blend / blend.sum()
        comp_idx = int(rng.choice(n, p=blend))

    # Sample from the chosen component
    mu_comp = gmm.means_[comp_idx]              
    cov_comp = gmm.covariances_[comp_idx]        

    sample = rng.multivariate_normal(mu_comp, cov_comp)
    x = float(np.clip(sample[0], *PLATE_X_BOUNDS))
    z = float(np.clip(sample[1], *PLATE_Z_BOUNDS))
    return x, z


# ---- Targeting bias helpers ----

def _push_sign(x: float, rng: np.random.Generator) -> float:
    """Choose a stable-ish direction to push horizontally toward an edge.
    If x already has a sign, keep it; otherwise pick a random side."""
    if x > 1e-6:
        return 1.0
    if x < -1e-6:
        return -1.0
    return 1.0 if rng.random() < 0.5 else -1.0


def apply_targeting_bias(
    mean_x: float,
    mean_z: float,
    pitch_type: str,
    balls: int | None,
    strikes: int | None,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Heuristic 'intent' adjustment.

    Location model tends to regress toward average/strike-ish means.
    This nudges the *mean* based on count + pitch type to get more realistic
    zone-edge and chase behavior.

    Returns: (biased_mean_x, biased_mean_z)
    """
    code = (pitch_type or "").upper().strip()

    # Default targets
    Z_CENTER = 2.5

    # Pitch-type bias strengths (feet)
    # (Breaking balls tend to live closer to edges / lower.)
    PT_EDGE = {
        "FF": 0.05,
        "FT": 0.06,
        "SI": 0.07,
        "FC": 0.07,
        "FS": 0.07,
        "SF": 0.07,
        "CH": 0.08,
        "SL": 0.10,
        "ST": 0.10,
        "CU": 0.11,
        "KC": 0.11,
    }
    PT_DROP_2STR = {
        # extra vertical drop with 2 strikes
        "FF": 0.12,
        "FT": 0.12,
        "SI": 0.13,
        "FC": 0.13,
        "FS": 0.13,
        "SF": 0.13,
        "CH": 0.16,
        "SL": 0.20,
        "ST": 0.20,
        "CU": 0.24,
        "KC": 0.24,
    }

    # Some pitchers intentionally "climb the ladder" with hard stuff on 2 strikes.
    PT_RISE_2STR = {
        "FF": 0.40,
        "FT": 0.35,
        "SI": 0.35,
        "FC": 0.38,
    }
    PT_RISE_2STR_PROB = {
        # probability of choosing the elevated target on 2 strikes
        "FF": 0.28,
        "FT": 0.18,
        "SI": 0.15,
        "FC": 0.22,
    }

    edge = PT_EDGE.get(code, 0.07)

    b = 0 if balls is None else int(balls)
    s = 0 if strikes is None else int(strikes)

    # 3-ball counts: attack zone more 
    if b >= 3:
        mean_x = 0.65 * mean_x  # closer to middle
        mean_z = 0.75 * mean_z + 0.25 * Z_CENTER

    # 2-strike counts: expand 
    if s >= 2:
        sign = _push_sign(mean_x, rng)
        mean_x = mean_x + sign * edge

        # With hard pitches, occasionally go upstairs instead of always burying.
        rise_prob = PT_RISE_2STR_PROB.get(code, 0.0)
        if rise_prob > 0 and rng.random() < rise_prob and b <= 1:
            mean_z = mean_z + PT_RISE_2STR.get(code, 0.35)
        else:
            mean_z = mean_z - PT_DROP_2STR.get(code, 0.16)

    # Even in neutral counts, gently encourage edges for breakers (and a bit for cutters)
    if b <= 1 and s <= 1 and code in {"SL", "ST", "CU", "KC", "CH", "FC"}:
        sign = _push_sign(mean_x, rng)
        # cutters get a smaller edge nudge than true breakers
        mult = 0.35 if code == "FC" else 0.5
        mean_x = mean_x + sign * (mult * edge)

    # Clamp to physical-ish bounds
    mean_x = float(np.clip(mean_x, PLATE_X_BOUNDS[0], PLATE_X_BOUNDS[1]))
    mean_z = float(np.clip(mean_z, PLATE_Z_BOUNDS[0], PLATE_Z_BOUNDS[1]))

    return mean_x, mean_z

def _sample_location(mean_x: float, mean_z: float, pitch_type: str, rng: np.random.Generator | None = None):
    """Legacy fallback sampler (used only if no GMM dict is available).
    Kept for backward compatibility; prefer sample_location_gmm() instead.
    """
    rng = rng or np.random.default_rng()
    # Use realistic std devs – wider than the original values to avoid mean-collapse.
    FALLBACK_STD = {"x": 0.70, "z": 0.75}
    x = float(np.clip(rng.normal(mean_x, FALLBACK_STD["x"]), *PLATE_X_BOUNDS))
    z = float(np.clip(rng.normal(mean_z, FALLBACK_STD["z"]), *PLATE_Z_BOUNDS))
    return x, z


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
    # Accept both naming conventions from the frontend
    last_pitch_type = user_context.get("last_pitch_type")
    if last_pitch_type is None:
        last_pitch_type = user_context.get("previous_pitch")
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

    # Accept both naming conventions from the frontend
    last_zone = user_context.get("last_zone")
    if last_zone is None:
        last_zone = user_context.get("previous_zone")

    if last_zone is not None:
        if "previous_zone" in w.columns:
            w.at[last_idx, "previous_zone"] = last_zone

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
    sample_location: bool = True,
    rng_seed: int | None = None,
    gmm_dict: dict | None = None,
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

    # --- Sanity checks: location artifacts MUST match the saved location model ---
    # location_model expects (None, seq_len, N)
    expected_loc_nf = None
    try:
        expected_loc_nf = int(location_model.input_shape[0][-1])
    except Exception:
        expected_loc_nf = None

    scaler_loc_nf = getattr(loc_scaler_X, "n_features_in_", None)
    features_loc_nf = len(loc_features)

    if expected_loc_nf is not None:
        if scaler_loc_nf is not None and int(scaler_loc_nf) != expected_loc_nf:
            raise ValueError(
                f"Location scaler_X expects {scaler_loc_nf} features but location_model expects {expected_loc_nf}. "
                "Your LOC_DIR artifacts are out of sync. Re-copy *all* location artifacts together "
                "(pitch_location_model.keras, features.pkl, scaler_X.pkl, scaler_Y.pkl) from the same training run "
                "into the folder your Flask app loads (LOC_DIR), then restart Flask."
            )
        if features_loc_nf != expected_loc_nf:
            raise ValueError(
                f"loc_features has {features_loc_nf} columns but location_model expects {expected_loc_nf}. "
                "This also indicates mixed artifacts. Re-copy the full set of location artifacts from the same run."
            )

    # Build feature matrices aligned to training features
    sub_pt  = ensure_columns(serving_window.copy(), pt_features)
    sub_loc = ensure_columns(serving_window.copy(), loc_features)

    Xpt  = pt_scaler_X.transform(sub_pt.values)[np.newaxis, :, :]
    Xloc = loc_scaler_X.transform(sub_loc.values)[np.newaxis, :, :]

    # Embedding IDs from encoders 
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
    allowed = artifacts.get("repertoire_map", {}).get(pitcher_mlbam)

    if allowed:
        mask = np.array([
            1.0 if cls in allowed else 0.0
            for cls in pt_label_enc.classes_
        ], dtype=np.float32)

        pt_probs = pt_probs * mask

        if pt_probs.sum() > 0:
            pt_probs = pt_probs / pt_probs.sum()
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

    loc_out_scaled = location_model.predict(
        [Xloc, p_ids[:, 0], b_ids[:, 0], pt_for_loc[np.newaxis, :]],
        verbose=0,
    )[0]

    # We now expect the location model to output 4 values:
    #   [mu_x, mu_z, log_var_x, log_var_z] in *scaled Y space*
    if loc_out_scaled.shape[-1] != 4:
        raise ValueError(
            f"Location model output has shape {loc_out_scaled.shape}; expected 4 (mu_x, mu_z, log_var_x, log_var_z). "
            "Did you forget to load the new Dense(4) location model?"
        )

    mu_scaled = np.array([loc_out_scaled[0], loc_out_scaled[1]], dtype=np.float32)

    # Convert mean back to original plate_x/plate_z units
    mean_xz = loc_scaler_Y.inverse_transform(mu_scaled.reshape(1, 2))[0]
    mean_x = float(mean_xz[0])
    mean_z = float(mean_xz[1])

    # Apply a light "intent" bias so means aren't always middle/middle.
    # This uses count + pitch type to push to edges or attack zone.
    last_row = serving_window.iloc[-1]
    balls = None
    strikes = None
    try:
        if "balls" in last_row.index:
            balls = int(last_row["balls"])
        if "strikes" in last_row.index:
            strikes = int(last_row["strikes"])
    except Exception:
        balls = None
        strikes = None

    bias_rng = np.random.default_rng(rng_seed)
    mean_x, mean_z = apply_targeting_bias(mean_x, mean_z, str(pt_pred), balls, strikes, bias_rng)

    # --- Sample final location ---
    # Prefer GMM-based sampling (realistic spread from real Statcast data).
    # Falls back to the legacy Gaussian sampler if no GMM dict is provided.
    if sample_location:
        rng = np.random.default_rng(rng_seed)
        _gmm_dict = gmm_dict if gmm_dict is not None else artifacts.get("location_gmm")
        if _gmm_dict:
            samp_x, samp_z = sample_location_gmm(
                mean_x, mean_z, str(pt_pred), _gmm_dict, rng,
                component_temperature=0.35,  # blend: model guidance + real mixture spread
            )
        else:
            # Legacy fallback
            samp_x, samp_z = _sample_location(mean_x, mean_z, str(pt_pred), rng)
    else:
        samp_x, samp_z = mean_x, mean_z

    # Clamp to reasonable bounds
    samp_x = float(np.clip(samp_x, PLATE_X_BOUNDS[0], PLATE_X_BOUNDS[1]))
    samp_z = float(np.clip(samp_z, PLATE_Z_BOUNDS[0], PLATE_Z_BOUNDS[1]))

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
        "location": {"plate_x": float(samp_x), "plate_z": float(samp_z)},
        "location_mean": {"plate_x": float(mean_x), "plate_z": float(mean_z)},
        "context": context
    }