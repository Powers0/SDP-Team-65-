pitch_model_rf.py uses Statcast data from MLB to train a machine learning model that predicts the type of pitch thrown based on the in-game context and the previous pitch

Install dependencies:
    pip3 install pybaseball pandas numpy scikit-learn seaborn matplotlib

Data Collection:
    Fetches Statcast pitch-by-pitch data using pybaseball.statcast
    Range: April 1, 2024 - May 30, 2024

Feature Engineering:
    Contextual variables include
    balls, strikes, outs_when_up, inning
    Binary encodings for on_1b, on_2b, on_3b
    bat_score, fld_score, and a computed score_diff

Filtering Pitch Types:
    Uses the six common pitch types
    FF (Four seam)
    SL (Slider)
    SI (Sinker)
    CH (Changeup)
    CU (Curveball)
    FC (Cutter)

Preprocessing:
    Encodes pitch types using LabelEncoder
    Standardizes numeric features with StandardScaler
    Splits data into train (80%) and test (20%)

Model Training:
    Uses a RandomForestClassifier with 200 trees

Evaluation:
    Generates a confusion matrix
    Feature importance bar chart


Current Example Output:
    Loading Statcast data...
    This is a large query, it may take a moment to complete
    Training RandomForestClassifier...
    Training complete.

    Test Accuracy: 43.38%

    Classification Report:
              precision    recall  f1-score   support

          CH       0.26      0.18      0.21        68
          CU       0.08      0.03      0.05        29
          FC       0.43      0.29      0.35        51
          FF       0.50      0.61      0.55       227
          SI       0.46      0.49      0.47       125
          SL       0.36      0.35      0.35       127


Analysis on Current Example Output:
    Currently, fastballs are predicted most accurately, while Curveballs are least accurate
    


Feature Importance Plot Rankings from most important to least:
    inning
    prev_pitch
    score_diff
    bat_score
    balls
    fld_score
    outs_when_up
    strikes
    on_1b, on_2b, on_3b 


