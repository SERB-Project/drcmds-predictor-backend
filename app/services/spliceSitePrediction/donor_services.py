import numpy as np
from tensorflow.keras.models import load_model
from typing import List
from app.services.spliceSitePrediction.preprocess import extract_features_donor

# Load the trained model
MODEL_PATH = "app/models/spliceSitePrediction/donor.h5"
model = load_model(MODEL_PATH)

# Predict a single sequence
def predict_donor_sequence(sequence: str) -> str:
    features = extract_features_donor(sequence)
    input_data = np.array(features).reshape(1, 15, 5)  # Match donor model input shape
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return str(predicted_class)

# Predict from a list of sequences (e.g., from a file)
def predict_donor_file(file_content) -> List[dict]:
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

        predicted_disease = predict_donor_sequence(sequence)
        results.append({"sequence": sequence, "prediction": predicted_disease})

    return results
