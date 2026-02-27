import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from EVLA_architecture import build_ev_model

ART_PATH = "artifacts/"


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
    PT_train  = np.load(ART_PATH + "PT_train.npy")
    PT_test   = np.load(ART_PATH + "PT_test.npy")
    LOC_train = np.load(ART_PATH + "LOC_train.npy")
    LOC_test  = np.load(ART_PATH + "LOC_test.npy")
    CTX_train = np.load(ART_PATH + "CTX_train.npy")
    CTX_test  = np.load(ART_PATH + "CTX_test.npy")
    y_train   = np.load(ART_PATH + "y_train.npy")
    y_test    = np.load(ART_PATH + "y_test.npy")

    pitch_type_dim = PT_train.shape[1]   # 14
    loc_dim        = LOC_train.shape[1]  # 4
    ctx_dim        = CTX_train.shape[1]  # varies by handedness dummies present in data

    # Scale continuous inputs (LOC and CTX); pitch-type one-hots don't need scaling
    loc_scaler = StandardScaler()
    LOC_train_s = loc_scaler.fit_transform(LOC_train)
    LOC_test_s  = loc_scaler.transform(LOC_test)

    ctx_scaler = StandardScaler()
    CTX_train_s = ctx_scaler.fit_transform(CTX_train)
    CTX_test_s  = ctx_scaler.transform(CTX_test)

    # Scale targets — EV (~50-120 mph) and LA (~-90 to 90 deg) have very different ranges
    target_scaler = StandardScaler()
    y_train_s = target_scaler.fit_transform(y_train)
    y_test_s  = target_scaler.transform(y_test)

    pickle.dump(loc_scaler,    open(ART_PATH + "loc_scaler.pkl",    "wb"))
    pickle.dump(ctx_scaler,    open(ART_PATH + "ctx_scaler.pkl",    "wb"))
    pickle.dump(target_scaler, open(ART_PATH + "target_scaler.pkl", "wb"))

    # Build and compile
    model = build_ev_model(pitch_type_dim=pitch_type_dim, loc_dim=loc_dim, ctx_dim=ctx_dim)
    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        # Reduce LR when val_loss plateaus — helpful for regression
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        [PT_train, LOC_train_s, CTX_train_s],
        y_train_s,
        validation_split=0.2,
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate in real units
    print("\nEvaluating on test set...")
    y_pred_s = model.predict([PT_test, LOC_test_s, CTX_test_s], batch_size=256, verbose=0)
    evaluate(y_test_s, y_pred_s, target_scaler)

    # Save
    model.save(ART_PATH + "ev_model.keras")
    print("\nSaved model:", ART_PATH + "ev_model.keras")