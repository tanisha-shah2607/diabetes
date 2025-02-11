import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from model import DiseasePredictionModel # type: ignore

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "disease_prediction_secret_key"  # Required for flash messages

# Initialize the model
model = DiseasePredictionModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = {
            'age': float(request.form['age']),
            'temperature': float(request.form['temperature']),
            'heart_rate': float(request.form['heart_rate']),
            'blood_pressure_systolic': float(request.form['blood_pressure_systolic']),
            'blood_pressure_diastolic': float(request.form['blood_pressure_diastolic']),
            'respiratory_rate': float(request.form['respiratory_rate'])
        }
        
        # Validate input ranges
        if not (0 <= features['age'] <= 120):
            flash('Please enter a valid age between 0 and 120')
            return redirect(url_for('index'))
        
        if not (35 <= features['temperature'] <= 42):
            flash('Please enter a valid temperature between 35°C and 42°C')
            return redirect(url_for('index'))
        
        # Make prediction
        prediction, confidence = model.predict(features)
        
        return render_template('result.html', 
                             prediction=prediction,
                             confidence=confidence,
                             features=features)
                             
    except ValueError as e:
        logger.error(f"Value error in prediction: {str(e)}")
        flash('Please enter valid numerical values for all fields')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        flash('An error occurred during prediction. Please try again.')
        return redirect(url_for('index'))
