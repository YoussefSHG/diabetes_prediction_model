from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # <-- This enables CORS for all routes

model = joblib.load('Model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received JSON:", data)

    # Ensure the keys are ordered like the model expects
    feature_order = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
                     'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
                     'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
                     'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
                     'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

    try:
        input_features = [float(data[key]) for key in feature_order]
    except KeyError as e:
        return jsonify({"error": f"Missing input key: {e}"}), 400

    prediction = model.predict([input_features])
    return jsonify({'prediction': prediction.tolist()})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
