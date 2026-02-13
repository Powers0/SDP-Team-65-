import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from swingtake_architecture import build_swingtake_model

ART_PATH = "artifacts/"


PITCHLOC_ART_DIR = "../Pitch Location Prediction/artifacts/"
PITCHLOC_MODEL_PATH = "../Pitch Location Prediction/artifacts/pitch_location_model.keras"

def engineered_features_from_loc(loc_xy):
    """
    loc_xy: (N,2) with [plate_x, plate_z] in real units (after inverse scaling)
    Returns extra features (N, extra_dim)
    """
    x = loc_xy[:, 0]
    z = loc_xy[:, 1]

    # geometry features
    dist_to_center = np.sqrt((x - 0.0) ** 2 + (z - 2.5) ** 2)

    # Simple strike indicator using a rough MLB strike zone
    # might replace  with batter-normalized zone later.
    is_strike = ((np.abs(x) <= 0.83) & (z >= 1.5) & (z <= 3.5)).astype(float)

    return np.column_stack([x, z, dist_to_center, is_strike])


if __name__ == "__main__":
    print("Loading Swing/Take dataset artifacts...")
    X_train = np.load(ART_PATH + "X_train.npy")
    X_test  = np.load(ART_PATH + "X_test.npy")
    P_train = np.load(ART_PATH + "P_train.npy")
    P_test  = np.load(ART_PATH + "P_test.npy")
    B_train = np.load(ART_PATH + "B_train.npy")
    B_test  = np.load(ART_PATH + "B_test.npy")
    PT_train = np.load(ART_PATH + "PT_train.npy")
    PT_test  = np.load(ART_PATH + "PT_test.npy")
    y_train = np.load(ART_PATH + "y_train.npy")
    y_test  = np.load(ART_PATH + "y_test.npy")

    pitchtype_dim = PT_train.shape[1]

    print("Loading pitch-location model + scaler_Y...")
    loc_model = load_model(PITCHLOC_MODEL_PATH)
    scaler_Y = pickle.load(open(PITCHLOC_ART_DIR + "scaler_Y.pkl", "rb"))


    """SHARED_DIR = "artifacts/"  # adjust if needed
    pitcher_le = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
    batter_le  = pickle.load(open(SHARED_DIR + "batter_le.pkl", "rb"))

    # If they look like raw MLBAM ids, encode them
    if P_train.max() >= loc_model.get_layer("embedding").input_dim:
        P_train = pitcher_le.transform(P_train.astype(int)).astype("int32")
        P_test  = pitcher_le.transform(P_test.astype(int)).astype("int32")

    if B_train.max() >= loc_model.get_layer("embedding_1").input_dim:
        B_train = batter_le.transform(B_train.astype(int)).astype("int32")
        B_test  = batter_le.transform(B_test.astype(int)).astype("int32")"""


    # Get predicted locations from pitch-location model
    print("Predicting locations (train)...")
    loc_train_scaled = loc_model.predict([X_train, P_train, B_train, PT_train], batch_size=256, verbose=1)
    print("Predicting locations (test)...")
    loc_test_scaled  = loc_model.predict([X_test,  P_test,  B_test,  PT_test],  batch_size=256, verbose=1)

    # inverse-transform to real plate_x and plate_z
    loc_train = scaler_Y.inverse_transform(loc_train_scaled)
    loc_test  = scaler_Y.inverse_transform(loc_test_scaled)

    # build extra features from predicted location
    extra_train = engineered_features_from_loc(loc_train)
    extra_test  = engineered_features_from_loc(loc_test)
    extra_dim = extra_train.shape[1]

    # scale extra features for the swing model and fit on train only 
    extra_scaler = StandardScaler()
    extra_train_s = extra_scaler.fit_transform(extra_train)
    extra_test_s  = extra_scaler.transform(extra_test)

    pickle.dump(extra_scaler, open(ART + "extra_scaler.pkl", "wb"))

    # build and train swing/take model
    model = build_swingtake_model(pitch_type_dim=pitchtype_dim, extra_dim=extra_dim)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        [PT_train, extra_train_s],
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=[early],
        verbose=1
    )

    # Evaluate (might add more metrics down here later)
    print("\nEvaluating...")
    probs = model.predict([PT_test, extra_test_s], batch_size=256, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # save model
    model.save("swingtake_model.keras")
    print("\nSaved model: swingtake_model.keras")
