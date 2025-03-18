import numpy as np

def preprocess_sequence(sequence: str):
    """
    Preprocess a given sequence for CNN model:
    - Remove unwanted characters
    - Pad/truncate to 30255 length
    - One-hot encode

    Returns:
        np.array: Encoded sequence (4 x 30255)
    """
    sequence = sequence.upper().replace("\n", "").replace(" ", "")
    sequence = "".join([c if c in "ACGTN" else "N" for c in sequence])

    if len(sequence) < 30255:
        sequence += "N" * (30255 - len(sequence))
    elif len(sequence) > 30255:
        sequence = sequence[:30255]

    mapping = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1], "N": [0, 0, 0, 0]}
    encoded = np.array([mapping[base] for base in sequence], dtype=np.float32).T  # (4, 30255)
    return encoded
