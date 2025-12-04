Pitch Location Prediction — Model Overview & Pipeline

This module trains a deep-learning model to predict the next pitch’s location (plate_x, plate_z) using Statcast pitch-by-pitch data and the output probabilities from the pitch-type model.
The model captures pitcher/batter tendencies, sequencing context, scored state, and the likelihood of pitch types to determine where the next pitch is expected to be thrown.

Dataset Builder (build_pitchlocation_dataset.py)

    Loads Statcast CSV files (2021–2024)

    Each file contains pitch-by-pitch tracking data including:
        •	pitch location
        •	count
        •	base state
        •	game score
        •	pitcher & batter IDs
        •	metadata (zone, game_pk, etc.)

    Preprocessing
        •	Drop rows missing required features
        •	Convert base occupancy (on_1b, on_2b, on_3b) to 0/1
        •	Compute score_diff
        •	One-hot encode pitcher & batter handedness (stand_*, p_throws_*)
        •	Convert pitcher/batter MLBAM IDs → category IDs (pitcher_id, batter_id)
        •	Sort pitches chronologically
        •	Extract features matrix X and target matrix Y = [plate_x, plate_z]

    Sequence Construction

    Using a sliding window:
        past 5 pitches (features) → next pitch's (plate_x, plate_z)

    For each sequence we also save:
	•	pitcher_id for that sample
	•	batter_id
	•	pitch_type probability vector from the pitch-type model

    Scaling
        •	scaler_X.fit() only on training data
        •	scaler_Y.fit() only on training target

    Artifacts Saved

    Everything required for inference is saved into artifacts/, including:
        •	scaled datasets
        •	scalers
        •	feature list
        •	pitch-type probability sequences
        •	embeddings category mappings (encoded inside the dataset)


Model Architecture pitchlocation_architecture.py
    The pitch-location model predicts the continuous 2D coordinates of the next pitch.
    The architecture uses both sequential and categorical information:

    Inputs
        1.	Pitch sequences
    (batch, SEQ_LEN=5, num_features)
        2.	Pitcher ID
    (batch,)
        3.	Batter ID
    (batch,)
        4.	Pitch-type probabilities
    (batch, n_classes) softmax vector

    Embedding Layers
        pitcher_emb = Embedding(num_pitchers, 8)
        batter_emb  = Embedding(num_batters, 8)

    These 8-dimensional vectors capture:
	•	pitcher’s habitual locations
	•	pitcher’s pitch tunneling patterns
	•	batter’s hot/cold zones
	•	batter’s susceptibility based on handedness

    They act as latent profile vectors for both participants in an at-bat.

    Main Processing Path
    The sequential features (X) go through:
        LSTM(128)
        Dense(64, relu)
        Dense(32, relu)
        Dense(2) → (plate_x, plate_z)
    
    LSTM(128)
    Dense(64, relu)
    Dense(32, relu)
    Dense(2) → (plate_x, plate_z)

Training Script train_pitchlocation_model.py