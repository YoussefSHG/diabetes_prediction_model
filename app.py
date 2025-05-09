from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(_name_)
CORS(app, resources={r"/": {"origins": ""}})  # Allow any origin

model = joblib.load('Model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = data['input']
    prediction = model.predict([input_features])
    return jsonify({'prediction': prediction.tolist()})

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
