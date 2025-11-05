Overview

    This script downloads (or loads cached) Statcast pitch-by-pitch CSV data for specified seasons, pre-processes it into fixed-length pitch sequences, builds an LSTM model augmented with learned embeddings for pitchers and batters, trains the model to predict pitch type, and evaluates the result (accuracy, classification report, confusion matrix, and training plots)


Features

    'balls', 'strikes', 'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b', 'bat_score', 'fld_score', 'zone', 'stand', 'p_throws'


Common Pitch Types Used
    ['FF', 'SL', 'SI', 'CH', 'CU', 'FC', 'ST']


Encoding and Standardization
    Handedness (stand and p_throws) one hot encoded
    previous_pitch one hot encoded
    previous_zone (previous pitch location zone) kept as numeric (with -1 for missing) then standardized


Sequence Creation
    For each plate appearance (grouped by pitcher, game_pk, at_bat_number), sliding-window sequences are created of length sequence_length (default 5).
	Each training sample uses the prior sequence_length timesteps (features + pitcher_id sequence + batter_id sequence) to predict the next pitch type.
	Split sequences into training (80%) and test (20%) partitions.
	Model architecture:


Model Architecture
    Inputs: input_features (sequence_length × feature_dim), input_pitcher (sequence of pitcher ids), input_batter (sequence of batter ids).
	Embeddings: small learned embeddings per pitcher and batter (embedding_dim=8).
	Concatenate features and embeddings along the last axis for each timestep.
	Masking layer applied (mask value 0.0).
	Single LSTM layer (128 units) followed by Dropout and Dense layers.
	Softmax final classification over len(le_pitch.classes_) pitch classes.
	Train with categorical crossentropy, adam optimizer, early stopping on validation loss


Evaluation
    test-set accuracy, classification report, confusion matrix, and training/validation accuracy plots.


Design Decisions
    Embedding dimension = 8. Small embedding dimension balances representational capacity with overfitting risk. 8 is a reasonable default given likely pitcher/batter counts and dataset size—you can increase with more data or tune via validation.


Architecture choices
	LSTM(128) — adequate capacity for learning medium-length dependencies; something in the 64–256 range is typical. 128 balances learning capacity and training speed.
	Dropout (0.3) — mitigates overfitting on dense layers and recurrent outputs.
	Dense(64) before final softmax helps combine the LSTM summary into discriminative representations.
	Categorical crossentropy with softmax is the correct loss for multi-class pitch-type classification.


Training choices
	EarlyStopping on validation loss with restore_best_weights=True prevents overfitting and selects the best epoch observed on validation data.
	Validation split within fit (0.2) means the training set has an internal validation set for early stopping/hyperparameter signal.



Sample Output
Evaluating model...
Test Accuracy: 52.69%
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step 

Classification Report:
              precision    recall  f1-score   support

          CH       0.31      0.23      0.26        57
          CU       0.17      0.04      0.06        25
          FC       0.56      0.45      0.50        42
          FF       0.55      0.66      0.60       252
          SI       0.67      0.67      0.67       121
          SL       0.48      0.38      0.43       121
          ST       0.38      0.50      0.43        50

    accuracy                           0.53       668
   macro avg       0.45      0.42      0.42       668
weighted avg       0.51      0.53      0.51       668


Possible Improvements:
    Fix scaler leakage: Fit StandardScaler only on training data.
    Use time-based validation: Train on older seasons and evaluate on the newest season to better estimate generalization
    Class balancing / focal loss / class weights: Mitigate imbalance.
    Set seeds for numpy and tensorflow for reproducibility (np.random.seed(42) and tf.random.set_seed(42)) and ensure deterministic GPU config if needed
    Add previous pitch velocity feature
    Add pitch count as a feature
    Current AB vs Entire Game History vs Game History Between Pitcher and Batter, try modeling just one pitcher and batter 



