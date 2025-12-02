from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# load model once
#model = load("pitch_model.joblib")

"""
HITTERS = ["Aaron Judge", "Juan Soto", "Mookie Betts"]
PITCHERS = ["Gerrit Cole", "Max Scherzer", "Zack Wheeler"]"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        hitter = request.form["hitter"]
        pitcher = request.form["pitcher"]

        # turn hitter/pitcher into model inputs

        X = build_features(hitter, pitcher)  

        # run prediction
        #prediction = model.predict([X])[0]
    return render_template("index.html")
    #return render_template("index.html", hitters=HITTERS, pitchers=PITCHERS, prediction=prediction)

def build_features(hitter, pitcher):

    features = []


    #features.append(HITTERS.index(hitter))
    #features.append(PITCHERS.index(pitcher))

    return np.array(features)

if __name__ == "__main__":
    app.run(debug=True)
