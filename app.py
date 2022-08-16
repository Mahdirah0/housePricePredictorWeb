from flask import Flask, render_template, redirect, url_for, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html", prediction=0)


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = str(round(model.predict(features)[0], 2))

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
