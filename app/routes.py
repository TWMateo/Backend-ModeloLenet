# app/routes.py

from flask import render_template, request, jsonify
from app import app
import tensorflow as tf
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from flask_cors import CORS
CORS(app)

model = load_model('models/modelo_cnn.h5')  

@app.route('/')
def index():
    return 'Api Modelo Telnet'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (28, 28))
        normalized_image = resized_image / 255.0
        input_image = np.reshape(normalized_image, (1, 28, 28, 1))

        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction)

        # Devolver la predicción como JSON
        return jsonify({'predicción': str(predicted_class)})

