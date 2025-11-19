pitch_location.py

This project predicts the plate_x and plate_z (horizontal and vertical location) of an MLB pitch using
    Game context features
    Batter and pitcher embeddings
    An LSTM sequence of past pitches
    Precomputed pitch-type probability vectors from a separate pitch-type model

The result is a modular system where pitch type and pitch location models work together for an at bat simulator


Model Architecture
    Sequence Input:
        8 pitch sliding window
        Standardized numeric features (balls, strikes, inning, score, base occupancy, handedness)
        Pitcher embedding
            Learned 32-dim embedding
        Batter embedding
            Learned 16-dim embedding
        Pitch Type Probabilities
            Precomputed softmax output from your pitch-type model
            Shape: (num sequences, num_pitch_types)

Neural Model Structure
    Sequence Input -> LSTM(128) -> LSTM(64) -> Dense(32)
    Pitcher Embedding -> RepeatVector -> concat
    Batter Embedding -> RepeatVector -> concat
    Pitch-Type Probability Vector -> concat
    -> Dense(2) -> Output: [plate_x, plate_z]
    Training Loss: MSE 
    Metrics: MAE

Dataset:
    Seasons 2022-2024

Results
    Test MSE: 0.9381
MAE: 0.7605

MAE plate_x: 0.738
MAE plate_z: 0.783

RMSE plate_x: 0.949
RMSE plate_z: 0.988




Model Type: 2 layer LSTM
    128 -> 64
    Early layers learn broad, high - dimensional patterns from raw inputs
    Later layers condense those into more specific, task-relevant features
    Encode -> compress

Pitcher and Batter embeddings (32 and 16) help map each ID to a learnable vector
Pitchers get a bigger number (32) because pitchers influence way more variability in the pitch sequence

Train = 85% of all generated sequences
Test = 15% of all generated sequences
Splitting happening on sequence examples ^ 

Recurrent Dropout = 0.1 (inside LSTM)
Regular Dropout 0.2 (between stacked layers)

10% of the recurrent connections are randomly dropped during training to stabilize LSTM training and prevent the model from memorizing entire sequences
After the first LSTM finishes processing all time steps, 20% of its outputs are zeroed out to prevent the model from relying too heavily on specific neurons