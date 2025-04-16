import numpy as np
from tensorflow.keras.models import load_model
from typing import List
from app.services.spliceSitePrediction.preprocess import extract_features_acceptor

# Load the trained model
MODEL_PATH = "app/models/spliceSitePrediction/acceptor.h5"
model = load_model(MODEL_PATH)

# Predict a single sequence
def predict_acceptor_sequence(sequence: str) -> str:
    features = extract_features_acceptor(sequence)
    input_data = np.array(features).reshape(1, 90, 5)  # Match model input shape
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return str(predicted_class)


# Predict from a file of sequences (list of strings)
def predict_acceptor_file(file_content):
    import csv
    from io import StringIO

    reader = csv.reader(StringIO(file_content))
    results = []

    for row in reader:
        if not row:
            continue  # skip empty rows
        sequence = row[0].strip()
        if not sequence:
            continue  # skip blank lines

        predicted_disease = predict_acceptor_sequence(sequence)
        results.append({"sequence": sequence, "prediction": predicted_disease})

    return results
