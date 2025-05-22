import numpy as np
import pandas as pd
import re

def clean_sequence(seq: str) -> str:
    """Remove unwanted characters and whitespace."""
    return re.sub(r'\s+', '', seq.upper())


def extract_features_acceptor(sequence: str) -> np.ndarray:
    # Define the one-hot encoding mapping
    mapping = {
        'A': [1, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0],
        'G': [0, 0, 1, 0, 0],
        'T': [0, 0, 0, 1, 0],
        'N': [0, 0, 0, 0, 1]  # unknown or padding
    }

    # Pad or trim sequence to 90
    seq = sequence.upper()
    seq = seq[:90].ljust(90, 'N')

    # Convert to one-hot encoding
    encoded = np.array([mapping.get(nt, mapping['N']) for nt in seq])

    return encoded

def extract_features_donor(sequence: str) -> np.ndarray:
    mapping = {
        'A': [1, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0],
        'G': [0, 0, 1, 0, 0],
        'T': [0, 0, 0, 1, 0],
        'N': [0, 0, 0, 0, 1]
    }
    seq = sequence.upper()
    seq = seq[:15].ljust(15, 'N')  # Trim or pad to length 15 for donor model
    encoded = np.array([mapping.get(nt, mapping['N']) for nt in seq])
    return encoded
