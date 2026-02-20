# Pitch Type Prediction

Predicts the next pitch type a pitcher will throw given the last 5 pitches of context, the current game situation, and learned profiles of both the pitcher and batter.

---

## Files

| File | Purpose |
|---|---|
| `build_pitchtype_dataset.py` | Loads raw Statcast CSVs, preprocesses, builds sequences, saves artifacts |
| `pitchtype_architecture.py` | Keras model definition |
| `train_pitchtype_model.py` | Trains the model, evaluates, saves `.keras` + `pitch_type_probs.npy` |
| `artifacts/` | All saved artifacts (model, scaler, label encoder, feature list, datasets) |

---

## Dataset Builder

**Input:** Raw Statcast CSVs (2021–2024), one per year.

**Preprocessing steps:**
- Drop rows with missing required features
- Convert base occupancy (`on_1b`, `on_2b`, `on_3b`) to binary 0/1
- Compute `score_diff = bat_score - fld_score`
- One-hot encode batter stance (`stand_L`, `stand_R`) and pitcher hand (`p_throws_L`, `p_throws_R`)
- One-hot encode previous pitch type (`previous_pitch_FF`, `previous_pitch_SL`, ...)
- Encode pitcher and batter MLBAM IDs as integer category codes

**Sequence construction:**

Each training sample is a sliding window of 5 consecutive pitches from the same at-bat:
```
[pitch_t-4, pitch_t-3, pitch_t-2, pitch_t-1, pitch_t] → pitch_type_{t+1}
```

**Artifacts saved to `artifacts/`:**
- `features.pkl` — ordered list of feature column names
- `scaler.pkl` — `StandardScaler` fit on training data only
- `label_encoder.pkl` — maps pitch type strings (FF, SL, ...) to integer class indices
- `processed_X.npy`, `processed_Y.npy` — full scaled dataset
- `processed_pitcher.npy`, `processed_batter.npy` — integer ID sequences
- `pitch_type_probs.npy` — softmax probability vectors for every sample (consumed by the location model)

---

## Model Architecture

```
Inputs:
  X          → (batch, 5, num_features)   # sequence of pitch features
  pitcher_id → (batch, 5)                 # pitcher MLBAM encoded ID, repeated per timestep
  batter_id  → (batch, 5)                 # batter MLBAM encoded ID, repeated per timestep

Embedding(num_pitchers, 8)  → pitcher_emb  (batch, 5, 8)
Embedding(num_batters,  8)  → batter_emb   (batch, 5, 8)

Concatenate([X, pitcher_emb, batter_emb])  → (batch, 5, num_features+16)
LSTM(128)
Dropout(0.3)
Dense(64, relu)
Dense(n_classes, softmax)                  → pitch type probability distribution
```

### Why these choices?

**Embeddings (dim=8):** Each pitcher and batter gets a learned 8-dimensional latent vector. This captures tendencies like pitch repertoire, sequencing habits, and handedness matchup effects without requiring hand-crafted features. 8 dimensions is large enough to encode meaningful patterns but small enough to avoid overfitting on less-common players.

**LSTM(128):** Pitch sequencing has temporal dependencies — what was thrown 2 pitches ago affects what comes next. The LSTM reads the 5-pitch window and produces a hidden state encoding that sequential context. 128 units balances capacity vs. training speed.

**Dropout(0.3):** Prevents the model from memorizing pitcher-specific sequences from training data, which would hurt generalization to counts and situations not well-represented in the data.

**Softmax output:** The model outputs a full probability distribution over all pitch types. This lets us both take the argmax for a deterministic prediction and sample stochastically for realism in the simulator. The probability vector is also passed to the location model as a conditioning signal.

---

## Training

- **Loss:** Categorical cross-entropy
- **Optimizer:** Adam (default lr)
- **Epochs:** Up to 12, with `EarlyStopping(patience=3)` on `val_loss`
- **Split:** 80/20 train/test, with 20% of train used for validation
- **Batch size:** 64

---

## Performance

Typical evaluation metrics (7-class pitch set):

| Metric | Value |
|---|---|
| Test Accuracy | ~52–53% |

**Context:** Predicting pitch type in baseball is inherently noisy — even MLB scouts with full game film and advance reports don't achieve much higher accuracy. The model performs well on high-frequency pitch types (FF, SI, FC) and shows lower performance on rarer or more situational pitches (CU, ST) due to class imbalance in the data.

---

## Artifacts Consumed Downstream

`pitch_type_probs.npy` is saved after training and used as an input feature to the **Pitch Location model** — the location model is conditioned on what pitch type is being thrown, which significantly improves its location predictions.
