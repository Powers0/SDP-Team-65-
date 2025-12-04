
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from pitchlocation_architecture import build_pitch_location_model

ART = "artifacts/"

print("Loading artifacts...")

X_train = np.load(ART + "X_train.npy")
X_test  = np.load(ART + "X_test.npy")
Y_train = np.load(ART + "Y_train.npy")
Y_test  = np.load(ART + "Y_test.npy")

P_train = np.load(ART + "P_train.npy")
P_test  = np.load(ART + "P_test.npy")
B_train = np.load(ART + "B_train.npy")
B_test  = np.load(ART + "B_test.npy")
PT_train = np.load(ART + "PT_train.npy")
PT_test  = np.load(ART + "PT_test.npy")

features = pickle.load(open(ART + "features.pkl", "rb"))
scaler_Y = pickle.load(open(ART + "scaler_Y.pkl", "rb"))

num_features = len(features)
seq_len = X_train.shape[1]
pitchtype_dim = PT_train.shape[1]
num_pitchers = max(P_train.max(), P_test.max()) + 1
num_batters = max(B_train.max(), B_test.max()) + 1

print("Dataset Loaded:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("Y_train:", Y_train.shape, "Y_test:", Y_test.shape)
print("P_train:", P_train.shape, "P_test:", P_test.shape)
print("B_train:", B_train.shape, "B_test:", B_test.shape)
print("PT_train:", PT_train.shape, "PT_test:", PT_test.shape)

model = build_pitch_location_model(
    seq_len=seq_len,
    num_features=num_features,
    num_pitchers=num_pitchers,
    num_batters=num_batters,
    pitch_type_dim=pitchtype_dim
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    [X_train, P_train, B_train, PT_train],
    Y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=256,
    callbacks=[early],
    verbose=1
)

print("\nEvaluating model...")
loss, mae = model.evaluate([X_test, P_test, B_test, PT_test], Y_test, verbose=2)
print(f"Test Loss (scaled): {loss:.4f}, Test MAE (scaled): {mae:.4f}")

Y_pred_scaled = model.predict([X_test, P_test, B_test, PT_test])

Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_true = scaler_Y.inverse_transform(Y_test)

mse = np.mean((Y_pred - Y_true)**2)
mae = np.mean(np.abs(Y_pred - Y_true))
rmse = np.sqrt(mse)

mae_x = mean_absolute_error(Y_true[:,0], Y_pred[:,0])
mae_z = mean_absolute_error(Y_true[:,1], Y_pred[:,1])
rmse_x = np.sqrt(mean_squared_error(Y_true[:,0], Y_pred[:,0]))
rmse_z = np.sqrt(mean_squared_error(Y_true[:,1], Y_pred[:,1]))

print("\n========== FULL METRICS ==========")
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"MAE plate_x: {mae_x:.3f}")
print(f"MAE plate_z: {mae_z:.3f}")
print(f"RMSE plate_x: {rmse_x:.3f}")
print(f"RMSE plate_z: {rmse_z:.3f}")

print("\n========== Example Predictions ==========")
for i in range(5):
    print(f"pred: {Y_pred[i]}    true: {Y_true[i]}")

plt.figure(figsize=(6,6))
plt.scatter(Y_true[:,0], Y_true[:,1], alpha=0.3, s=10, label="True")
plt.scatter(Y_pred[:,0], Y_pred[:,1], alpha=0.3, s=10, label="Predicted")
plt.xlabel("plate_x")
plt.ylabel("plate_z")
plt.legend()
plt.title("Pitch Location True vs Predicted")
plt.tight_layout()
plt.show()

model.save("pitch_location_model.keras")
print("\nSaved model: pitch_location_model.keras")