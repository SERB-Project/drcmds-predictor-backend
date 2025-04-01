import torch
import numpy as np
import os
import json
from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient
from app.models.sarsVariantsClassification_MutationAnalysis.cnn_model import InterSSPPCNN
from app.services.sarsVariantsClassificationMutation.predict import classify_variant as predict

variant_to_idx = {
    "B.1.1.7": 0,
    "B.1.351": 1,
    "P.1": 2,
    "B.1.617.2": 3,
    "B.1.1.529": 4
}

# # Load model
model_path = "./app/models/sarsVariantsClassification_MutationAnalysis/sars_variant_classifier.pth"
input_length = 30255
model = InterSSPPCNN(input_length)
model.load_state_dict(torch.load(model_path))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Load reference sequence
reference_path = "./app/models/sarsVariantsClassification_MutationAnalysis/encoded_reference_sequence.npy"
reference_sequence = np.load(reference_path)

# Define explainability composite
composite = EpsilonGammaBox(epsilon=1e-4, gamma=0.5, low=-1.0, high=1.0)



def explain_nucleotide_mutations(sequence, top_n=15):
    """
    Explain nucleotide-wise mutations for a given sequence.
    Returns top N mutations with reference and mutated nucleotides.
    """
    # Convert sequence to tensor
    # sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

    
    # Get prediction
    predicted_label = predict(sequence)  # Use predict function from predict.py
    # print(predicted_label)
    predicted_label = variant_to_idx[predicted_label]
    
    # Compute attribution 
    with Gradient(model=model, composite=composite) as attributor:
        outputs, relevance = attributor(
            sequence_tensor.permute(0, 2, 1), 
            torch.eye(5).to(device)[predicted_label].unsqueeze(0)  # Fix here!
        )

    summed_relevance = relevance[0].sum(axis=0)
    
    # Get top N mutations
    top_positions = np.argsort(-summed_relevance.cpu().detach().numpy())[:top_n]
    mutations = []
    
    for pos in top_positions:
        base_encoding = sequence[:, pos]  # Get one-hot encoded nucleotide
        ref_encoding = reference_sequence[:, pos]  # Get reference one-hot encoding

        ref_nucleotide = get_nucleotide_from_encoding(ref_encoding)
        mutated_nucleotide = get_nucleotide_from_encoding(base_encoding)

        if ref_nucleotide != mutated_nucleotide:
            mutations.append({
                "Position": int(pos),
                "Reference": ref_nucleotide,
                "Mutated": mutated_nucleotide
            })
    
    return mutations

def get_nucleotide_from_encoding(encoding):
    """ Convert one-hot encoding to nucleotide (A, T, C, G) """
    bases = ['A', 'T', 'C', 'G']
    return bases[np.argmax(encoding)] if np.sum(encoding) > 0 else 'N'
