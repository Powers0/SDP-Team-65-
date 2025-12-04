# pitchlocation_architecture.py

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, RepeatVector
from tensorflow.keras.models import Model

def build_pitch_location_model(seq_len, num_features, num_pitchers, num_batters, pitch_type_dim):

    seq_input = Input(shape=(seq_len, num_features))
    pitcher_input = Input(shape=(), dtype="int32")
    batter_input = Input(shape=(), dtype="int32")
    pitchtype_input = Input(shape=(pitch_type_dim,))

    pitcher_emb = Embedding(num_pitchers, 32)(pitcher_input)
    batter_emb  = Embedding(num_batters, 16)(batter_input)

    p_rep = RepeatVector(seq_len)(pitcher_emb)
    b_rep = RepeatVector(seq_len)(batter_emb)

    x = Concatenate(axis=-1)([seq_input, p_rep, b_rep])
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dense(32, activation="relu")(x)

    x = Concatenate()([x, pitchtype_input])
    out = Dense(2)(x)

    return Model(
        inputs=[seq_input, pitcher_input, batter_input, pitchtype_input],
        outputs=out
    )