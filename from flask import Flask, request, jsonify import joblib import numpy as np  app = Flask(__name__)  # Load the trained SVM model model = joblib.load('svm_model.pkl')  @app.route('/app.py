from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load('svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = data['input']  # expects a list of features
    prediction = model.predict([input_features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
