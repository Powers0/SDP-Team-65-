from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    Concatenate, RepeatVector, BatchNormalization
)
from tensorflow.keras.models import Model


def build_ev_model(
    seq_len: int,
    num_features: int,
    num_pitchers: int,
    num_batters: int,
    pitch_type_dim: int,
    loc_dim: int,
):
    """
    Predicts exit velocity and launch angle for balls in play,
    using pitch sequence context via LSTM.

    Inputs:
      - seq_input:       (seq_len, num_features)   sliding window of prior pitch context
      - pitcher_input:   ()                         encoded pitcher ID
      - batter_input:    ()                         encoded batter ID
      - pitchtype_input: (pitch_type_dim,)          one-hot of the TARGET pitch type
      - loc_input:       (loc_dim,)                 plate_x, plate_z, dist_to_center, is_strike

    Output:
      - (2,) — [exit_velocity, launch_angle] in scaled units
    """
    # --- Sequence branch (mirrors pitch location model) ---
    seq_input     = Input(shape=(seq_len, num_features), name="seq_input")
    pitcher_input = Input(shape=(),                      name="pitcher_input", dtype="int32")
    batter_input  = Input(shape=(),                      name="batter_input",  dtype="int32")

    # Embeddings: pitcher gets more capacity than batter (more influence on contact outcomes)
    pitcher_emb = Embedding(num_pitchers, 32, name="pitcher_emb")(pitcher_input)
    batter_emb  = Embedding(num_batters,  32, name="batter_emb")(batter_input)
    # Batter gets a larger embedding than in the location model — who's hitting matters a lot for EV/LA
    # (pitcher got 32, batter got 16 in location model; both get 32 here)

    # Tile embeddings across timesteps so they can be concatenated with the sequence
    p_rep = RepeatVector(seq_len)(pitcher_emb)
    b_rep = RepeatVector(seq_len)(batter_emb)

    x = Concatenate(axis=-1)([seq_input, p_rep, b_rep])

    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.20)(x)
    x = LSTM(64)(x)
    x = Dropout(0.20)(x)
    x = Dense(32, activation="relu")(x)

    # --- Target pitch branch ---
    # Concatenate LSTM output with the actual pitch type + location of the ball in play
    pitchtype_input = Input(shape=(pitch_type_dim,), name="pitchtype_input")
    loc_input       = Input(shape=(loc_dim,),        name="loc_input")

    merged = Concatenate()([x, pitchtype_input, loc_input])

    merged = Dense(64, activation="relu")(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.20)(merged)

    merged = Dense(32, activation="relu")(merged)
    merged = Dropout(0.10)(merged)

    # Linear activation — LA can be negative
    out = Dense(2, activation="linear", name="ev_la")(merged)

    return Model(
        inputs=[seq_input, pitcher_input, batter_input, pitchtype_input, loc_input],
        outputs=out
    )