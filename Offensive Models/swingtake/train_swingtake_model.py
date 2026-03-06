import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from swingtake_architecture import build_swingtake_model

ART_PATH = "artifacts/"


if __name__ == "__main__":
    print("Loading Swing/Take dataset artifacts...")
    PT_train  = np.load(ART_PATH + "PT_train.npy")
    PT_test   = np.load(ART_PATH + "PT_test.npy")
    LOC_train = np.load(ART_PATH + "LOC_train.npy")
    LOC_test  = np.load(ART_PATH + "LOC_test.npy")
    y_train   = np.load(ART_PATH + "y_train.npy")
    y_test    = np.load(ART_PATH + "y_test.npy")

    pitchtype_dim = PT_train.shape[1]   # len(ALL_PITCH_TYPES), e.g. 14
    extra_dim     = LOC_train.shape[1]  # 4: plate_x, plate_z, dist, is_strike

    # Scale the location features (pitch-type one-hots don't need scaling)
    loc_scaler = StandardScaler()
    LOC_train_s = loc_scaler.fit_transform(LOC_train)
    LOC_test_s  = loc_scaler.transform(LOC_test)

    pickle.dump(loc_scaler, open(ART_PATH + "loc_scaler.pkl", "wb"))

    # Build and train swing/take model
    model = build_swingtake_model(pitch_type_dim=pitchtype_dim, extra_dim=extra_dim)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        [PT_train, LOC_train_s],
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=[early],
        verbose=1
    )

    # Evaluate
    print("\nEvaluating...")
    probs = model.predict([PT_test, LOC_test_s], batch_size=256, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save model
    model.save(ART_PATH + "swingtake_model.keras")
    print("\nSaved model:", ART_PATH + "swingtake_model.keras")
