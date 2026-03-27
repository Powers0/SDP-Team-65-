from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_contact_model(pitch_type_dim: int, loc_dim: int, ctx_dim: int):
    pitchtype_input = Input(shape=(pitch_type_dim,), name="pitchtype_onehot")
    location_input  = Input(shape=(loc_dim,),        name="location_features")
    context_input   = Input(shape=(ctx_dim,),         name="context_features")

    x = Concatenate()([pitchtype_input, location_input, context_input])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.30)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.20)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.10)(x)
    out = Dense(3, activation="softmax")(x)

    return Model(inputs=[pitchtype_input, location_input, context_input], outputs=out)