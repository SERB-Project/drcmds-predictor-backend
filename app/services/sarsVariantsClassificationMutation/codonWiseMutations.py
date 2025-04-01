import torch
import numpy as np
import os
from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient
from app.models.sarsVariantsClassification_MutationAnalysis.cnn_model import InterSSPPCNN
from app.services.sarsVariantsClassificationMutation.predict import classify_variant as predict

# Variant mapping
variant_to_idx = {
    "B.1.1.7": 0,
    "B.1.351": 1,
    "P.1": 2,
    "B.1.617.2": 3,
    "B.1.1.529": 4
}

# Load model
model_path = "./app/models/sarsVariantsClassification_MutationAnalysis/sars_variant_classifier.pth"
input_length = 30255
model = InterSSPPCNN(input_length)
model.load_state_dict(torch.load(model_path))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load reference sequence
reference_path = "./app/models/sarsVariantsClassification_MutationAnalysis/encoded_reference_sequence.npy"
reference_sequence = np.load(reference_path)

# Explainability composite
composite = EpsilonGammaBox(epsilon=1e-4, gamma=0.5, low=-1.0, high=1.0)

def explain_codon_mutations(sequence, top_n=15):
    """
    Explain codon-wise mutations for a given sequence.
    Returns top N codon mutations with reference and mutated codons.
    """
    # Convert sequence to tensor
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

    # Get prediction
    predicted_label = predict(sequence)  
    predicted_label = variant_to_idx[predicted_label]
    
    # Compute attribution
    with Gradient(model=model, composite=composite) as attributor:
        outputs, relevance = attributor(
            sequence_tensor.permute(0, 2, 1), 
            torch.eye(5).to(device)[predicted_label].unsqueeze(0)
        )

    summed_relevance = relevance[0].sum(axis=0).cpu().detach().numpy()

    # Get top N relevant positions
    top_positions = np.argsort(-summed_relevance)[:top_n]

    mutations = []
    checked_codons = set()  # Track codons to avoid duplicates

    for pos in top_positions:
        codon_start = pos - (pos % 3)  # Align to the first base of the codon

        if codon_start in checked_codons:
            continue  # Skip if we've already checked this codon
        checked_codons.add(codon_start)

        # Extract codons
        ref_codon = extract_codon(reference_sequence, codon_start)
        mutated_codon = extract_codon(sequence, codon_start)

        # Only record mutations
        if ref_codon != mutated_codon:
            mutations.append({
                "Codon_Position": int(codon_start),  # Ensure Python int
                "Reference_Codon": str(ref_codon),   # Ensure string
                "Mutated_Codon": str(mutated_codon)  # Ensure string
            })

    
    return mutations

def extract_codon(sequence, start_pos):
    """
    Extracts a codon (3 nucleotides) from a one-hot encoded sequence.
    
    Args:
        sequence (np.array): One-hot encoded sequence.
        start_pos (int): Start position of the codon.
    
    Returns:
        str: Extracted codon sequence (3 nucleotides).
    """
    bases = ['A', 'T', 'C', 'G']
    codon = ""

    for i in range(3):
        pos = start_pos + i
        if pos < sequence.shape[1]:  # Ensure within bounds
            nucleotide_encoding = sequence[:, pos]
            nucleotide = bases[np.argmax(nucleotide_encoding)] if np.sum(nucleotide_encoding) > 0 else 'N'
            codon += nucleotide
        else:
            codon += 'N'  # Pad with 'N' if out of bounds

    return codon
