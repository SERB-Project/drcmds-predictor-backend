import torch
import numpy as np
from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient
from Bio.Data import CodonTable
from Bio.Seq import Seq

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

# Codon table
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]

def explain_codon_mutations(sequence, top_n=15):
    """
    Explain codon-wise mutations and classify them as silent, missense, or nonsense.
    """
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

    predicted_label = predict(sequence)
    predicted_label = variant_to_idx[predicted_label]

    with Gradient(model=model, composite=composite) as attributor:
        outputs, relevance = attributor(
            sequence_tensor.permute(0, 2, 1),
            torch.eye(5).to(device)[predicted_label].unsqueeze(0)
        )

    summed_relevance = relevance[0].sum(axis=0).cpu().detach().numpy()
    top_positions = np.argsort(-summed_relevance)[:top_n]

    mutations = []
    checked_codons = set()

    for pos in top_positions:
        codon_start = pos - (pos % 3)
        if codon_start in checked_codons:
            continue
        checked_codons.add(codon_start)

        ref_codon = extract_codon(reference_sequence, codon_start)
        mutated_codon = extract_codon(sequence, codon_start)

        if ref_codon != mutated_codon:
            mutation_type = classify_codon_mutation(ref_codon, mutated_codon)
            mutations.append({
                "Codon_Position": int(codon_start),
                "Reference_Codon": str(ref_codon),
                "Mutated_Codon": str(mutated_codon),
                "Mutation_Type": mutation_type
            })

    return mutations

def extract_codon(sequence, start_pos):
    """Extract codon from one-hot sequence."""
    bases = ['A', 'T', 'C', 'G']
    codon = ""

    for i in range(3):
        pos = start_pos + i
        if pos < sequence.shape[1]:
            encoding = sequence[:, pos]
            base = bases[np.argmax(encoding)] if np.sum(encoding) > 0 else 'N'
            codon += base
        else:
            codon += 'N'
    return codon

def classify_codon_mutation(ref_codon, mutated_codon):
    """
    Classify mutation as Silent, Missense, or Nonsense based on amino acid change.
    """
    try:
        if 'N' in ref_codon or 'N' in mutated_codon:
            return "Unknown"

        ref_aa = str(Seq(ref_codon).translate(table=standard_table))
        mut_aa = str(Seq(mutated_codon).translate(table=standard_table))

        if ref_aa == mut_aa:
            return "Silent"
        elif mut_aa == "*":
            return "Nonsense"
        else:
            return "Missense"
    except Exception as e:
        return "Unknown"
