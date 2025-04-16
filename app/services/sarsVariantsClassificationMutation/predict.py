import torch
import numpy as np
from app.models.sarsVariantsClassification_MutationAnalysis.cnn_model import InterSSPPCNN

# Define model path
MODEL_PATH = "app/models/sarsVariantsClassification_MutationAnalysis/sars_variant_classifier.pth"
INPUT_LENGTH = 30255  # Expected length of input sequence

# Mapping indices to variant names
idx_to_variant = {
    0: "B.1.1.7",
    1: "B.1.351",
    2: "P.1",
    3: "B.1.617.2",
    4: "B.1.1.529"
}

# Load the trained model
model = InterSSPPCNN(INPUT_LENGTH)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))  # Ensure compatibility with CPU & GPU
model.eval()  # Set model to evaluation mode

# Move model to appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_variant(encoded_sequence: np.array) -> str:
    """
    Predicts the SARS-CoV-2 variant from an encoded sequence.

    Args:
        encoded_sequence (np.array): A one-hot encoded sequence of shape (4, sequence_length).

    Returns:
        str: Predicted variant name.
    """
    # Ensure the input sequence has the correct shape: (4, sequence_length)
    if encoded_sequence.shape != (4, INPUT_LENGTH):
        raise ValueError(f"Expected encoded sequence of shape (4, {INPUT_LENGTH}), but got {encoded_sequence.shape}")

    # Convert to PyTorch tensor and ensure correct shape: (1, 4, sequence_length)
    sequence_tensor = torch.tensor(encoded_sequence, dtype=torch.float32).unsqueeze(0)  # (1, 4, seq_length)

    # Compute mask (Ensure correct shape)
    mask = (sequence_tensor.sum(dim=1, keepdim=True) != 0).float()  # (1, 1, seq_length)

    print(f"Input shape before model: {sequence_tensor.shape}")  # Debug print
    print(f"Mask shape before model: {mask.shape}")  # Debug print

    # Move tensors to the correct device
    sequence_tensor = sequence_tensor.to(device)
    mask = mask.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(sequence_tensor)  # Mask is handled inside model
        predicted_idx = torch.argmax(output, dim=1).item()  # Get predicted class index

    # Return human-readable variant name
    return idx_to_variant.get(predicted_idx, "Unknown Variant")
