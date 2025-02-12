from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib  # For loading the trained model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [float(request.form[feature]) for feature in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction']]
        
        # Convert to NumPy array and reshape
        input_data = np.array(data).reshape(1, -1)
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Return result
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
