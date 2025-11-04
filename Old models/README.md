pitch_model.py

 
Uses MLB Statcase data from Pybaseball to build a pitch type prediction model.
Given contextual features, the model attempts to predicts the type of pitch (fastball, curveball, changeup, etc)

Data collected using Pybaseball Library:
from pybaseball import statcast
df = statcast(start_dt='2024-04-01', end_dt='2024-04-30')

Dependencies
pytorch, pybaseball, pandas, skikit-learn, numpy, seaborn

Model Design
A Multi-Layer Perceptron (MLP) with learned batter and pitcher embeddings

Training
Optimized with adam and class weights

Evaluation
Includes F1-Score, ROC-AUC, Top-k accuracy, and Confusion matrix

Data Cleaning and Feature Selection
Relevant game context features are used, like balls, strikes, outs, inning, base runners on, score differential
Base occupancy converted to binary flags
Certain pitch types like pitchouts, knuckle curves, euphus pitches, are excluded due to rarity and inconsistency

Encoding and Normalization
Categorical variables like p_throws, stand (batter stance), inning_topbot are label-encoded
Continuous features are standardized using StandardScaler to help gradient-based training converge faster

Batter and Pitcher Emeddings
Each batter and pitcher is assigned a small learnable vector (embedding)

Weighting
compute_class_weight calculates weights inversely proportional to pith frequency
Loss function CrossEntropyLoss incorporates hese weights to attempt to avvoid bias towards common pitches
Attempting to balance rare pitch types like Splitters and Sweepers

Current Limitations and Things to Add
Sequencing: The model currently treats each pitch as independent
Relatively limited data timeframe sample (2 months)
Likely aiming to add an RNN, LSTM, or Transformer model to implement pitch sequencing
Looking to include pitch location