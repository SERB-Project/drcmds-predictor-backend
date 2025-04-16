import numpy as np

def pad_sequences(sequences, max_length=11195):
    return [seq.ljust(max_length, 'N')[:max_length] for seq in sequences]

def one_hot_encoding(seq):
    base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([base_dict.get(base, [0, 0, 0, 0]) for base in seq])
