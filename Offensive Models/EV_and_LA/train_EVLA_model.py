import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from EVLA_architecture import build_ev_model

ART_PATH = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"


def evaluate(y_true, y_pred, target_scaler):
    """Print per-target MAE in real units after inverse-transforming predictions."""
    y_true_real = target_scaler.inverse_transform(y_true)
    y_pred_real = target_scaler.inverse_transform(y_pred)

    mae_ev = np.mean(np.abs(y_true_real[:, 0] - y_pred_real[:, 0]))
    mae_la = np.mean(np.abs(y_true_real[:, 1] - y_pred_real[:, 1]))

    print(f"  Exit Velocity MAE : {mae_ev:.2f} mph")
    print(f"  Launch Angle MAE  : {mae_la:.2f} degrees")
    return mae_ev, mae_la


if __name__ == "__main__":
    print("Loading EV/LA dataset artifacts...")
    X_train   = np.load(ART_PATH + "X_train.npy")
    X_test    = np.load(ART_PATH + "X_test.npy")
    P_train   = np.load(ART_PATH + "P_train.npy")
    P_test    = np.load(ART_PATH + "P_test.npy")
    B_train   = np.load(ART_PATH + "B_train.npy")
    B_test    = np.load(ART_PATH + "B_test.npy")
    PT_train  = np.load(ART_PATH + "PT_train.npy")
    PT_test   = np.load(ART_PATH + "PT_test.npy")
    LOC_train = np.load(ART_PATH + "LOC_train.npy")
    LOC_test  = np.load(ART_PATH + "LOC_test.npy")
    y_train   = np.load(ART_PATH + "y_train.npy")
    y_test    = np.load(ART_PATH + "y_test.npy")

    target_scaler = pickle.load(open(ART_PATH + "target_scaler.pkl", "rb"))

    seq_len        = X_train.shape[1]
    num_features   = X_train.shape[2]
    pitch_type_dim = PT_train.shape[1]
    loc_dim        = LOC_train.shape[1]

    # Derive encoder sizes from the shared label encoders (same source as dataset build)
    pitcher_le = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
    batter_le  = pickle.load(open(SHARED_DIR + "batter_le.pkl",  "rb"))
    num_pitchers = len(pitcher_le.classes_)
    num_batters  = len(batter_le.classes_)

    print(f"seq_len={seq_len}  num_features={num_features}  pitch_type_dim={pitch_type_dim}")
    print(f"loc_dim={loc_dim}  num_pitchers={num_pitchers}  num_batters={num_batters}")

    # Build and compile
    model = build_ev_model(
        seq_len=seq_len,
        num_features=num_features,
        num_pitchers=num_pitchers,
        num_batters=num_batters,
        pitch_type_dim=pitch_type_dim,
        loc_dim=loc_dim,
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        [X_train, P_train, B_train, PT_train, LOC_train],
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate in real units
    print("\nEvaluating on test set...")
    y_pred = model.predict(
        [X_test, P_test, B_test, PT_test, LOC_test],
        batch_size=256, verbose=0
    )
    evaluate(y_test, y_pred, target_scaler)

    # Save
    model.save(ART_PATH + "ev_model.keras")
    print("\nSaved model:", ART_PATH + "ev_model.keras")