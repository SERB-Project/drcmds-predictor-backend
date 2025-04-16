import os
import numpy as np
from tensorflow.keras.models import load_model
from app.services.viralDiseasePrediction.preprocess import pad_sequences, one_hot_encoding

# Load model
MODEL_PATH = "app/models/viralDiseasePrediction/cnn_model_all.h5"
model = load_model(MODEL_PATH)

# Disease mapping
disease_mapping = {0: "HBV", 1: "INFLUENZA", 2: "HCV", 3: "DENGUE"}

def predict_sequence(sequence: str):
    max_length = 11195
    padded_sequence = pad_sequences([sequence], max_length)
    encoded_sequence = one_hot_encoding(padded_sequence[0])
    input_data = np.array(encoded_sequence).reshape(1, max_length, 4)

    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    return disease_mapping.get(predicted_class, "Unknown")

def predict_file(file_content):
    import csv
    from io import StringIO

    reader = csv.reader(StringIO(file_content))
    results = []

    for row in reader:
        sequence = row[0]
        predicted_disease = predict_sequence(sequence)
        results.append({"sequence": sequence, "prediction": predicted_disease})

    return results
