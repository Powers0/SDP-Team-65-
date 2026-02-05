# Older Models Analysis

*   Random Forrest Model - Accuracy: 48.48%
    *   Dependancies: pybaseball pandas numpy scikit-learn seaborn matplotlib
    *   Features: Contextual variables include balls, strikes, outs\_when\_up, inning Binary encodings for on\_1b, on\_2b, on\_3b bat\_score, fld\_score, and a computed score\_diff, previous 3 pitch types
    *   Dataset: Fetches Statcast pitch-by-pitch data using pybaseball.statcast Range: April 1, 2024 - May 30, 2024
    *   Train/Test Split: Splits data into train (80%) and test (20%)
    *   Model Training: RandomForestClassifier with 200 trees
    *   Output: Confusion Matrix, Feature importance bar chart, Precision, Recall and F1-score table for each pitch, overall accuracy of model
![](https://t9017284666.p.clickup-attachments.com/t9017284666/6f8ad8b7-699c-4cec-b369-f8d5764ea9a4/image.png)
This model was one of the better preforming early models. Random forrest is strong with picking out trends in the data. However, random forrest does not have built in sequencing for the pitches. To accomplish this, 3 additional features were added; the type of the last 3 pitches. More features of the previous pitches could have been implemented but adding 15-20 more features would just overcomplicate and overfit the model negatively. Due to these limitations, we decided to move away from random forrest as preferred machine learning method.
*   Nueral Network Model - Accuracy: 12%
    *   Dependancies: pybaseball pandas numpy scikit-learn seaborn matplotlib, pytorch
    *   Features: Contextual variables include balls, strikes, outs\_when\_up, inning Binary encodings for on\_1b, on\_2b, on\_3b bat\_score, fld\_score, and a computed score\_diff
    *   Dataset: Fetches Statcast pitch-by-pitch data using pybaseball.statcast Range: April 1, 2024 - May 30, 2024
    *   Train/Test Split: Splits data into train (80%) and test (20%)
    *   Model Training: Neural Network with vector length of 64 and max epoch of 50
    *   Output: Confusion Matrix, Precision, Recall and F1-score table for each pitch, overall accuracy of model
![](https://t9017284666.p.clickup-attachments.com/t9017284666/3c7f1c5d-cb7c-4a30-a36a-f50bf3421715/image.png)
This model was the first use of a nueral network which would eventually lead us to our current models that use LSTM algorithms. Sequencing is a key factor in determining the next pitch during an at bat and a regular neural network does not retain memory. This means they treat each pitch independantly and do not account for the previous pitches which hindered the accuracy of the model. Nueral networks also have a sharp decay in accurancy over longer sequences of data which doesn't help improve the accuracy. LSTM models incorporate sequencing and use gated architechture to regulate the flow of data to combat the potential of accuracy decay.
*   Fastball Only Model - Accuracy: 31%
    *   Dependancies: pybaseball pandas numpy scikit-learn seaborn matplotlib, pytorch
    *   Features: Contextual variables include balls, strikes, outs\_when\_up, inning Binary encodings for on\_1b, on\_2b, on\_3b bat\_score, fld\_score, and a computed score\_diff
    *   Dataset: Fetches Statcast pitch-by-pitch data using pybaseball.statcast Range: April 1, 2024 - May 30, 2024
    *   Train/Test Split: Splits data into train (80%) and test (20%)
    *   Model Training: Neural Network with vector length of 64 and max epoch of 30
    *   Output: Confusion Matrix, Precision, Recall and F1-score table for each pitch, overall accuracy of model
![](https://t9017284666.p.clickup-attachments.com/t9017284666/00dd1388-8920-4bd7-a7a8-7dd59787619a/image.png)
This model was the first test model with the data set. It was a basic nueral network model that only tested fastballs. The model was used to ensure that it could meet the baseline accuracy of the dataset. "If it only predicted fastball would the model be accurate." test. The model was successful in determining that conclusion and did not mislabel any fastballs as another type of pitch.