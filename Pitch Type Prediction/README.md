This module trains a neural network model to predict the next pitch type in an MLB at-bat using Statcast pitch-by-pitch data.
The system uses LSTM sequence modeling, learned pitcher and batter embeddings, and a fully modular preprocessing + training pipeline.

build_pitchtypedata.py
    Applies preprocessing
	•	Converts baserunners → 0/1
	•	Computes score differential (score_diff)
	•	Fixes missing zones (zone=-1)
	•	One-hot encodes:
	•	batter stance (stand)
	•	pitcher throwing hand (p_throws)
	•	previous pitch type (previous_pitch)
	•	Computes previous pitch zone (previous_zone)
	•	Builds:
	•	pitcher_id = categorical integer code
	•	batter_id = categorical integer code

    Creates fixed-length sequences
    For each at-bat: [pitch_t-5, pitch_t-4, pitch_t-3, pitch_t-2, pitch_t-1] → pitch_type_t
    Saves reusable artifacts
	•	features.pkl (column order)
	•	scaler.pkl (fit ONLY on training data)
	•	label_encoder.pkl
	•	X_train / X_test
	•	pitcher/batter ID sequences
	•	pitchtype_model.keras (after training)

    These artifacts let the front-end run predictions without preprocessing the raw CSV data again.


train_pitchtypemodel.py
    Uses the artifact dataset and trains the LSTM + embedding model.

Key training features:
	•	Train/test split (80/20)
	•	Validation split (20%)
	•	EarlyStopping(patience=3)
	•	Accuracy, classification report, confusion matrix saved/displayed
	•	Softmax probability output saved to pitch_type_probs.npy for the pitch-location model


Model Architecture pitch_model_lstm.py
    This architecture was designed for interpretable sequential modeling while embedding characteristics of pitchers and batters.

Inputs
	1.	X → (batch, SEQ_LEN, num_features)
	2.	pitcher_id → (batch, SEQ_LEN)
	3.	batter_id → (batch, SEQ_LEN)

    Embeddings
    pitcher_emb = Embedding(num_pitchers, 8)
    batter_emb  = Embedding(num_batters, 8)
    Each pitcher/batter learns an 8-dimensional vector capturing tendencies such as:
	•	pitch repertoire & sequencing
	•	release consistency
	•	batter hot/cold zones
	•	handedness effects

    Concatenation

    At each timestep:
        [features | pitcher_emb | batter_emb]
    Masking(mask_value=0.0)
    LSTM(128)
    Dropout(0.3)
    Dense(64, relu)
    Dense(n_classes, softmax)

    Training
	•	Loss: categorical crossentropy
	•	Optimizer: Adam
	•	EarlyStopping: monitors val_loss

    Why these choices?
    Embedding(8)
    Large enough to encode tendencies, small enough to prevent overfitting
    LSTM(128)
    Balanced capacity for pitch sequencing patterns
    Dropout(0.3)
    Prevents overfitting in sequential model
    Dense(64)
    Intermediate abstraction for final classification
    Softmax
    Multi-class pitch prediction requires probability distribution

📊 Final Model Performance

Typical evaluation:
	•	Accuracy ~52–53% on real-world 7-class pitch set
	•	Good performance on FF, SI, FC
	•	Lower performance on CU, ST (class imbalance)

