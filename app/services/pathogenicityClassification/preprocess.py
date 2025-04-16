import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Bio import SeqIO

# Load reference genome
REFERENCE_GENOME_FILE = "app/models/pathogenicityClassification/gene.fna"
reference_genome = SeqIO.to_dict(SeqIO.parse(REFERENCE_GENOME_FILE, "fasta"))

# Label Encoder for base pair mutations
known_mutations = [
    "A>C", "A>G", "A>T", "C>A", "C>G", "C>T",
    "G>A", "G>C", "G>T", "T>A", "T>C", "T>G"
]
bp_encoder = LabelEncoder()
bp_encoder.fit(known_mutations)

# Mapping of molecular consequences to feature names
MC_LABELS = {
    "synonymous variant": "mc_synonymous_variant",
    "3 prime UTR variant": "mc_3_prime_UTR_variant",
    "5 prime UTR variant": "mc_5_prime_UTR_variant",
    "splice donor variant": "mc_splice_donor_variant",
    "splice acceptor variant": "mc_splice_acceptor_variant",
    "nonsense": "mc_nonsense",
    "intron variant": "mc_intron_variant",
    "missense variant": "mc_missense_variant",
    "stop lost": "mc_stop_lost"
}

# Expected feature order
EXPECTED_FEATURES = [
    "position", "BP_A>C", "BP_A>G", "BP_A>T", "BP_C>A", "BP_C>G", "BP_C>T",
    "BP_G>A", "BP_G>C", "BP_G>T", "BP_T>A", "BP_T>C", "BP_T>G",
    "mc_synonymous_variant", "mc_3_prime_UTR_variant", "mc_5_prime_UTR_variant",
    "mc_splice_donor_variant", "mc_splice_acceptor_variant", "mc_nonsense",
    "mc_intron_variant", "mc_missense_variant", "mc_stop_lost",
    "Prev_A", "Prev_C", "Prev_G", "Prev_T",
    "Next_A", "Next_C", "Next_G", "Next_T"
]

def get_prev_next_allele(chrom, position):
    seq = reference_genome.get(chrom)
    if seq:
        seq = seq.seq
        prev = seq[position - 2] if position > 1 else "N"
        next = seq[position] if position < len(seq) else "N"
        return prev, next
    return "N", "N"

def preprocess_input(spdi, consequence):
    chrom, pos, deleted, inserted = spdi.split(":")
    position = int(pos)
    bp_mutation = f"{deleted}>{inserted}"

    # Initialize all features to 0
    features = {feature: 0 for feature in EXPECTED_FEATURES}
    features["position"] = position

    # Encode mutation
    if bp_mutation in known_mutations:
        features[f"BP_{bp_mutation}"] = 1

    # Previous/Next Alleles
    prev, next_ = get_prev_next_allele(chrom, position)
    if prev in "ACGT":
        features[f"Prev_{prev}"] = 1
    if next_ in "ACGT":
        features[f"Next_{next_}"] = 1

    # Molecular consequence
    for label in consequence:
        if label in MC_LABELS:
            features[MC_LABELS[label]] = 1

    # Convert to DataFrame
    df = pd.DataFrame([features])
    df = df[EXPECTED_FEATURES]
    return df
