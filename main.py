from flask import Flask, request, render_template, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the model
model = load_model('predict_lung_cancer_model.keras')

# Load the scaler
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        gender = int(request.form['GENDER'])
        age = int(request.form['AGE'])
        smoking = int(request.form['SMOKING'])
        yellow_fingers = int(request.form['YELLOW_FINGERS'])
        anxiety = int(request.form['ANXIETY'])
        peer_pressure = int(request.form['PEER_PRESSURE'])
        chronic_disease = int(request.form['CHRONIC_DISEASE'])
        fatigue = int(request.form['FATIGUE'])
        allergy = int(request.form['ALLERGY'])
        wheezing = int(request.form['WHEEZING'])
        alcohol_consuming = int(request.form['ALCOHOL_CONSUMING'])
        coughing = int(request.form['COUGHING'])
        shortness_of_breath = int(request.form['SHORTNESS_OF_BREATH'])
        swallowing_difficulty = int(request.form['SWALLOWING_DIFFICULTY'])
        chest_pain = int(request.form['CHEST_PAIN'])

        # Create an array with the data
        input_data = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                                chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                                coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(input_data_scaled)
        prediction_percent = prediction[0][0] * 100

        return render_template('index.html', prediction_text=f'Probability of having lung cancer: {prediction_percent:.2f}%')

    except Exception as e:
        return str(e)

@app.route('/templates/<path:filename>')
def custom_static(filename):
    return send_from_directory('templates', filename)

if __name__ == '__main__':
    app.run(debug=True)
