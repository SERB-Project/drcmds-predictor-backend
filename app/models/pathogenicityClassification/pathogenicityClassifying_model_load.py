import os
import pickle

# Path to saved model
MODEL_PATH = "app/models/pathogenicityClassification/pathogenicityClassifier.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"🚨 Model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()