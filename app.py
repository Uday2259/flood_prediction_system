from flask import Flask, render_template, request
import numpy as np
import joblib

# Create Flask App
app = Flask(__name__)

# Load Saved Model and Scaler
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction Page
@app.route("/Predict")
def predict_page():
    return render_template("index.html")


# Prediction Function
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from form
        temp = float(request.form["Temp"])
        humidity = float(request.form["Humidity"])
        cloud_cover = float(request.form["Cloud_Cover"])
        annual = float(request.form["ANNUAL"])
        jan_feb = float(request.form["Jan_Feb"])
        mar_may = float(request.form["Mar_May"])
        jun_sep = float(request.form["Jun_Sep"])
        oct_dec = float(request.form["Oct_Dec"])
        avgjune = float(request.form["avgjune"])
        sub = float(request.form["sub"])

        # Create feature array
        data = np.array([[
            temp,
            humidity,
            cloud_cover,
            annual,
            jan_feb,
            mar_may,
            jun_sep,
            oct_dec,
            avgjune,
            sub
        ]])

        # Scale the input
        scaled_data = scaler.transform(data)

        # Predict
        prediction = model.predict(scaled_data)

        # Show Result
        if prediction[0] == 1:
            return render_template("chance.html")
        else:
            return render_template("no_chance.html")

    except Exception as e:
        return f"Error: {e}"


# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)