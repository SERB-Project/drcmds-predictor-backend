import torch
import os
from app.models.sarsVariantsClassification_MutationAnalysis.cnn_model import InterSSPPCNN 

MODEL_PATH = "app/models/sarsVariantsClassification_MutationAnalysis/sars_variant_classifier.pth"  # Path to saved model
INPUT_LENGTH = 30255  # Input length

def load_model():
    """
    Loads the trained CNN model from .pth file.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"🚨 Model not found at {MODEL_PATH}")

    model = InterSSPPCNN(input_length=INPUT_LENGTH)  # Initialize model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode
    return model

cnn_model = load_model()  # Load model once at startup
