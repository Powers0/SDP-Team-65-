# Pitch Location Prediction

Predicts where a pitch will cross home plate (`plate_x`, `plate_z` in feet) given the pitch sequence context, the pitcher/batter matchup, and what type of pitch is being thrown. Uses a two-stage approach: an LSTM trained with a probabilistic loss function to predict a mean location, followed by a Gaussian Mixture Model (GMM) fitted on real Statcast data to produce realistic final locations.

---

## Files

| File | Purpose |
|---|---|
| `build_pitchlocation_dataset.py` | Preprocesses Statcast data and builds training sequences |
| `pitchlocation_architecture.py` | Keras model definition |
| `train_pitchlocation_model.py` | Trains the model with Gaussian NLL loss, saves artifacts |
| `fit_location_gmm.py` | Fits a GMM per pitch type on real Statcast data, saves `location_gmm.pkl` |
| `artifacts/` | All saved artifacts (model, scalers, feature list, GMMs, datasets) |

---

## Dataset Builder

**Input:** Raw Statcast CSVs (2021–2024) + `pitch_type_probs.npy` from the pitch type model.

**Preprocessing steps:**
- Drop rows with missing `plate_x`, `plate_z`, or required features
- Convert base occupancy to binary 0/1
- Compute `score_diff`
- One-hot encode handedness and previous pitch type
- Encode pitcher/batter IDs as integer category codes

**Sequence construction:**

Same sliding window as the pitch type model — 5 pitches of context predicting the next pitch's location:
```
[pitch_t-4, ..., pitch_t] → (plate_x_{t+1}, plate_z_{t+1})
```

Each sample also includes:
- Pitcher and batter integer IDs
- The pitch type probability vector from the trained pitch type model (used as a conditioning input)

**Target scaling:** `plate_x` and `plate_z` are standardized with a `StandardScaler` fit on training data only. The scaler is saved as `scaler_Y.pkl` and used at inference to convert predictions back to feet.

---

## Model Architecture

```
Inputs:
  seq_input      → (batch, 5, num_features)    # pitch sequence features
  pitcher_input  → (batch,)                    # pitcher ID (single value)
  batter_input   → (batch,)                    # batter ID (single value)
  pitchtype_input → (batch, pitch_type_dim)    # pitch type probability vector

Embedding(num_pitchers, 32) → pitcher_emb  (batch, 32)
Embedding(num_batters,  16) → batter_emb   (batch, 16)

RepeatVector(5)(pitcher_emb) → (batch, 5, 32)   # broadcast across timesteps
RepeatVector(5)(batter_emb)  → (batch, 5, 16)

Concatenate([seq_input, pitcher_emb, batter_emb]) → (batch, 5, num_features+48)
LSTM(128, return_sequences=True)
Dropout(0.2)
LSTM(64)
Dense(32, relu)

Concatenate([dense_out, pitchtype_input])
Dense(4)  → [mu_x, mu_z, log_var_x, log_var_z]  (in scaled Y space)
```

### Why larger embeddings than the pitch type model?

The location model uses pitcher embeddings of dim=32 and batter embeddings of dim=16 (vs. dim=8 in the pitch type model). Location is more pitcher-specific than pitch type — a pitcher's tendencies to work inside vs. outside, or to live in the upper zone, are subtle patterns that benefit from a larger latent representation.

### Why output 4 values instead of 2?

A standard regression model would output `[mu_x, mu_z]` — a single point prediction. Instead the model outputs:

```
[mu_x, mu_z, log_var_x, log_var_z]
```

The extra two outputs are the **log-variance** of the model's uncertainty in each dimension. This enables training with a proper probabilistic loss (Gaussian NLL) and gives the model a way to express "I'm less certain about this location" for harder-to-predict counts or matchups.

---

## Loss Function: Gaussian Negative Log-Likelihood

Instead of MSE, the model is trained with **Gaussian NLL**:

```
L = 0.5 * sum_dim [ log_var + (y_true - mu)^2 / exp(log_var) ]
```

**Why not MSE?**

MSE implicitly assumes a fixed, equal uncertainty for every prediction. Gaussian NLL lets the model learn *per-prediction uncertainty* — it can output a wider variance when the situation is ambiguous (first pitch of an at-bat, 3-0 count) and a tighter variance when the pitcher's intent is clearer (2-strike chase pitch). This produces more realistic location distributions rather than collapsing everything toward the mean.

**Log-variance stability:** We predict `log(var)` rather than `var` directly. This keeps the value unconstrained (can be any real number) and avoids the numerical issues of predicting a strictly positive quantity. At inference, `exp(log_var)` recovers the variance. Log-variance is clipped to `[-7, 3]` during training to prevent loss explosions.

---

## GMM: Gaussian Mixture Model for Realistic Sampling

### Why a GMM on top of the LSTM?

The LSTM predicts a mean location that tends to regress toward the center of the strike zone — a known limitation of sequence-to-sequence regression models. Real pitchers don't throw everything down the middle; they work edges, bury breaking balls, and climb the ladder with fastballs.

To capture the true **shape** of a pitcher's location distribution, we fit a Gaussian Mixture Model (GMM) separately on the full 2021–2024 Statcast dataset, grouped by pitch type:

```
GMM[pitch_type] ~ GaussianMixture(
    n_components = 3-5,   # varies by pitch type (see N_COMPONENTS_OVERRIDE)
    covariance_type = "full",
    n_init = 5,
    random_state = 42
)
```

Each component captures a distinct "target region" a pitcher throws to — for example, a 4-seam fastball GMM might have components for: up-and-in, up-and-away, and middle-down.

### Component counts by pitch type

| Pitch | Components | Rationale |
|---|---|---|
| FF, FT, SI, FC, KC | 4 | Four quadrants of the zone |
| FS, SF, KN, EP | 3 | Simpler distribution (mostly down) |
| SL, ST, CU | 5 | Richer shape — these pitches work multiple edges |
| CH | 4 | Down + arm-side, two main target zones |

### Zone rates from the fitted GMMs (pure sampling)

| Pitch | Zone% |
|---|---|
| FF | 53.7% |
| SI | 54.8% |
| FC | 50.9% |
| SL | 46.8% |
| CU | 46.5% |
| CH | 40.6% |

These align well with real MLB averages.

---

## Inference: Combining LSTM Mean + GMM Sampling

At inference time, the pipeline is:

1. **LSTM predicts** `[mu_x, mu_z, log_var_x, log_var_z]` in scaled Y space
2. **Inverse-transform** the mean back to feet
3. **Apply targeting bias** — small heuristic nudges based on count + pitch type to correct for the model's tendency to predict near the center (e.g., 2-strike pitches pushed lower/to the edge)
4. **GMM sampling with `component_temperature=0.35`:**
   - Find the GMM component whose centroid is nearest to the LSTM's predicted mean (preserves directional intent)
   - Blend: 65% weight on nearest component, 35% on full mixture weights
   - Sample `(plate_x, plate_z)` from the selected component's full covariance Gaussian

**`component_temperature` explained:**

- `0.0` = always pick the single nearest GMM component → model's directional signal is fully preserved, but no mixture diversity (too many pitches cluster in one spot)
- `1.0` = sample purely from real mixture weights → fully realistic spread, but ignores what the LSTM predicted
- `0.35` = blends both: the model guides *where* in the zone, the GMM provides realistic *spread and shape*

---

## Artifacts

| Artifact | Description |
|---|---|
| `pitch_location_model.keras` | Trained Keras model |
| `features.pkl` | Ordered feature column list for inference |
| `scaler_X.pkl` | Input feature scaler |
| `scaler_Y.pkl` | Target (`plate_x`, `plate_z`) scaler — used to convert predictions back to feet |
| `location_gmm.pkl` | Dict of `{pitch_type: GaussianMixture}` — must be regenerated if CSV data changes |

---

## Performance

| Metric | Value |
|---|---|
| MAE plate_x | ~0.70 ft |
| MAE plate_z | ~0.76 ft |
| RMSE plate_x | ~0.88 ft |
| RMSE plate_z | ~0.95 ft |

These are point-estimate metrics on the model mean only. The actual sampling spread (via GMM) is intentionally wider to reflect real pitch-to-pitch variation.
