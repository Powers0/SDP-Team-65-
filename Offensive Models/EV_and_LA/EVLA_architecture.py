from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.models import Model


def build_ev_model(pitch_type_dim: int, loc_dim: int, ctx_dim: int):
    """
    Inputs:
      - pitchtype_input: (pitch_type_dim)  one-hot of actual Statcast pitch_type
      - loc_input:       (loc_dim)         plate_x, plate_z, dist_to_center, is_strike
      - ctx_input:       (ctx_dim)         count, handedness, game-state features

    Output:
      - (2,) — [exit_velocity, launch_angle] 
    """
    pitchtype_input = Input(shape=(pitch_type_dim,), name="pitchtype_onehot")
    loc_input       = Input(shape=(loc_dim,),        name="location_features")
    ctx_input       = Input(shape=(ctx_dim,),        name="context_features")

    x = Concatenate()([pitchtype_input, loc_input, ctx_input])

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.30)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.20)(x)

    x = Dense(32, activation="relu")(x)
    x = Dropout(0.10)(x)

    # Linear activation — targets are continuous and can be negative (LA ranges ~-90 to 90)
    out = Dense(2, activation="linear", name="ev_la")(x)

    return Model(inputs=[pitchtype_input, loc_input, ctx_input], outputs=out)