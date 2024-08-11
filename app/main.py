# app/main.py
from flask import Flask, send_from_directory, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from app import app
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model('app/sentiment_model.h5')
with open('app/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sequence = tokenizer.texts_to_sequences([data['text']])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return jsonify({'sentiment': sentiment})
