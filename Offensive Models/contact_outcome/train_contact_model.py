import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


from contact_architecture import build_contact_model

ART_PATH = "artifacts/"

if __name__ == "__main__":
    print("Loading contact outcome dataset artifacts...")
    PT_train  = np.load(ART_PATH + "PT_train.npy")
    PT_test   = np.load(ART_PATH + "PT_test.npy")
    LOC_train = np.load(ART_PATH + "LOC_train.npy")
    LOC_test  = np.load(ART_PATH + "LOC_test.npy")
    CTX_train = np.load(ART_PATH + "CTX_train.npy")
    CTX_test  = np.load(ART_PATH + "CTX_test.npy")
    y_train   = np.load(ART_PATH + "y_train.npy")
    y_test    = np.load(ART_PATH + "y_test.npy")
    P_train = np.load(ART_PATH + "P_train.npy")
    P_test  = np.load(ART_PATH + "P_test.npy")
    B_train = np.load(ART_PATH + "B_train.npy")
    B_test  = np.load(ART_PATH + "B_test.npy")


    pitchtype_dim = PT_train.shape[1]
    loc_dim       = LOC_train.shape[1]
    ctx_dim       = CTX_train.shape[1]

    loc_scaler = StandardScaler()
    LOC_train_s = loc_scaler.fit_transform(LOC_train)
    LOC_test_s  = loc_scaler.transform(LOC_test)
    pickle.dump(loc_scaler, open(ART_PATH + "loc_scaler.pkl", "wb"))

    CT_CONTINUOUS = ["balls", "strikes", "outs_when_up", "inning", "score_diff"]
    ctx_scaler = StandardScaler()
    CTX_train_s = CTX_train.copy()
    CTX_test_s  = CTX_test.copy()
    CTX_train_s[:, :5] = ctx_scaler.fit_transform(CTX_train[:, :5])
    CTX_test_s[:,  :5] = ctx_scaler.transform(CTX_test[:, :5])
    pickle.dump(ctx_scaler, open(ART_PATH + "ctx_scaler.pkl", "wb"))


    pitcher_le   = pickle.load(open("../../artifacts/shared/pitcher_le.pkl", "rb"))
    batter_le    = pickle.load(open("../../artifacts/shared/batter_le.pkl",  "rb"))
    num_pitchers = len(pitcher_le.classes_)
    num_batters  = len(batter_le.classes_)

    model = build_contact_model(
        pitch_type_dim=pitchtype_dim, loc_dim=loc_dim, ctx_dim=ctx_dim,
        num_pitchers=num_pitchers, num_batters=num_batters
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        [PT_train, LOC_train_s, CTX_train_s, P_train, B_train],
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=[early],
        verbose=1
)


    print("\nEvaluating...")
    probs = model.predict([PT_test, LOC_test_s, CTX_test_s, P_test, B_test], batch_size=256, verbose=0)

    preds = np.argmax(probs, axis=1)

    classes = pickle.load(open(ART_PATH + "classes.pkl", "rb"))
    print(classification_report(y_test, preds, target_names=classes, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    model.save(ART_PATH + "contact_model.keras")
    print("\nSaved model:", ART_PATH + "contact_model.keras")