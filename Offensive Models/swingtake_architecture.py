from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_swingtake_model(pitch_type_dim: int, extra_dim: int):
    """
    Inputs:
      - pitchtype_input: (pitch_type_dim)  7-dim 
      - extra_input:     (extra_dim)       predicted x,z, dist, strike_prob

    Output:
      - P(swing)
    """
    pitchtype_input = Input(shape=(pitch_type_dim,), name="pitchtype_probs")
    extra_input = Input(shape=(extra_dim,), name="extra_features")

    x = Concatenate()([pitchtype_input, extra_input])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.30)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.20)(x)
    out = Dense(1, activation="sigmoid")(x)

    return Model(inputs=[pitchtype_input, extra_input], outputs=out)
