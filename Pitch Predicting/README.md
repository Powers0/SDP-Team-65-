pitch_type_model.py

 
Uses MLB Statcase data from Pybaseball to build a pitch type prediction model.
Given contextual features like the count, pitcher's throwing hand, batter's stance, pitcher, batter, - the model attempts to predicts the type of pitch (fastball, curveball, changeup, etc)

This model is a Multi-Layer Perceptron (MLP), built using PyTorch. It perfoms multi-class classification to predict pitch type from game state and pitch level features
Part of larger object aiming to simulate entire at bats

Data collected using Pybaseball Library:
from pybaseball import statcast
df = statcast(start_dt='2024-04-01', end_dt='2024-04-30')

Dependencies
pytorch, pybaseball, pandas, skikit-learn, numpy, seaborn