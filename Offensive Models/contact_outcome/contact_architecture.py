from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Embedding, Flatten
from tensorflow.keras.models import Model

def build_contact_model(pitch_type_dim, loc_dim, ctx_dim, num_pitchers, num_batters, embed_dim=8):
    pitchtype_input = Input(shape=(pitch_type_dim,), name="pitchtype_onehot")
    location_input  = Input(shape=(loc_dim,),        name="location_features")
    context_input   = Input(shape=(ctx_dim,),         name="context_features")

    pitcher_input = Input(shape=(1,), name="pitcher_id")
    batter_input  = Input(shape=(1,), name="batter_id")

    p_emb = Flatten()(Embedding(num_pitchers, embed_dim)(pitcher_input))
    b_emb = Flatten()(Embedding(num_batters,  embed_dim)(batter_input))

    x = Concatenate()([pitchtype_input, location_input, context_input, p_emb, b_emb])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.30)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.20)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.10)(x)
    out = Dense(3, activation="softmax")(x)

    return Model(inputs=[pitchtype_input, location_input, context_input, pitcher_input, batter_input], outputs=out)
