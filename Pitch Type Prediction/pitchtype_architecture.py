from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_pitch_model(num_features, num_pitchers, num_batters, embed_dim=8, num_classes=7):
    f_in = Input(shape=(5, num_features))
    p_in = Input(shape=(5,))
    b_in = Input(shape=(5,))

    p_emb = Embedding(num_pitchers, embed_dim)(p_in)
    b_emb = Embedding(num_batters, embed_dim)(b_in)

    x = Concatenate()([f_in, p_emb, b_emb])
    x = LSTM(128)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=[f_in, p_in, b_in], outputs=out)