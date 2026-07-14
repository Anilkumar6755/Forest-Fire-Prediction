from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("forest_fire_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        temperature = float(request.form["temperature"])
        oxygen = float(request.form["oxygen"])
        humidity = float(request.form["humidity"])

        features = np.array([[temperature, oxygen, humidity]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "🔥 High Chance of Forest Fire"
        else:
            result = "🌳 Low Chance of Forest Fire"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=str(e))


if __name__ == "__main__":
    app.run(debug=True)