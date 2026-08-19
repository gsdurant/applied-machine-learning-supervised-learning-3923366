# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model pipeline
# This will load both the preprocessor and the classifier
try:
    # Adjust path if necessary, assuming model is in the same directory
    model_path = 'titanic_xgb_model.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please ensure it's saved.")
        model_pipeline = None
    else:
        model_pipeline = joblib.load(model_path)
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

# Define the features that the model expects, in the correct order
# These should match the 'features' list used during training
EXPECTED_FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

@app.route('/')
def home():
    """Simple root endpoint to confirm the API is running."""
    return "Welcome to the Titanic Survival Predictor API!"

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500

    print('In predict()')

    data = request.get_json(force=True)

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Convert input data to a pandas DataFrame
    # It's crucial that the column names match EXPECTED_FEATURES
    try:
        input_df = (pd.DataFrame([data], columns=EXPECTED_FEATURES)
        .astype({
            'embarked': 'category', 'sex': 'category'})
        )
    except KeyError as e:
        return jsonify({'error': f"Missing expected feature: {e}"}), 400
    except Exception as e:
        return jsonify({'error': f"Invalid input data format: {e}"}), 400

    print (input_df)
    print (input_df.dtypes)

    # Make prediction
    try:
        prediction = model_pipeline.predict(input_df)
        prediction_proba = model_pipeline.predict_proba(input_df)

        # The prediction is a numpy array, convert to list/int for JSON
        result = {
            'prediction': int(prediction[0]),
            'probability_not_survived': float(prediction_proba[0][0]),
            'probability_survived': float(prediction_proba[0][1])
        }
        return jsonify(result)
    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({'error': f"Prediction failed: {e}. Check server logs for details."}), 500

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, use a production-ready WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5555, debug=True)

