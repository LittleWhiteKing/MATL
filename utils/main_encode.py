import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import numpy as np

def one_hot(seq):

    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    encoded = np.zeros((4, len(seq)))

    for i, base in enumerate(seq):
        if base in mapping:
            encoded[mapping[base], i] = 1
        else:
            encoded[:, i] = 0
    return encoded


def two_mer_encode(seq):

    bases = ['A', 'C', 'G', 'T']

    two_mers = {a + b: i for i, (a, b) in enumerate((a, b) for a in bases for b in bases)}
    encoded = np.zeros((len(two_mers), len(seq) - 1), dtype=int)
    for i in range(len(seq) - 1):
        two_mer = seq[i:i + 2]
        if two_mer in two_mers:
            encoded[two_mers[two_mer], i] = 1
        else:
            encoded[:, i] = 0
    return encoded

def reverse_complement(seq):

    comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}

    return ''.join(comp.get(x, 'N') for x in reversed(seq))


def encode_sequence(seq):

    one_mer_original = one_hot(seq)
    rev_comp = reverse_complement(seq)
    one_mer_rev_comp = one_hot(rev_comp)
    two_mer_encoded = two_mer_encode(seq)
    final_matrix = np.zeros((20, 403))


    final_matrix[0:4, 0:101] = one_mer_original  #0-100
    final_matrix[4:20, 101:201] = two_mer_encoded
    final_matrix[0:4, 201:302] = one_mer_rev_comp
    final_matrix[0:4, 302:403] = one_mer_original
    # Fill 2-mer
    return final_matrix


class SampleReader:

    def __init__(self, file_name):
        """
            file_path:
                wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk
        """
        base_path = '/home/fanjinli/myproject/DNA/01myproject/final_project/dataset/' + file_name
        self.seq_path = os.path.join(base_path, 'sequence')
        self.shape_path = os.path.join(base_path, 'shape')
    def get_seq(self, Test=False):
        file_name = 'Test_seq.csv' if Test else 'Train_seq.csv'
        file_path = os.path.join(self.seq_path, file_name)
        row_seq = pd.read_csv(file_path, sep=',', header=None)
        seq_num = row_seq.shape[0]


        completed_seqs = np.empty((seq_num, 20, 403))
        completed_labels = np.empty((seq_num, 1))

        for i in range(seq_num):

            completed_seqs[i] = encode_sequence(row_seq.loc[i, 1])
            completed_labels[i] = row_seq.loc[i, 2]
        return completed_seqs, completed_labels

    def get_shape(self, shapes, Test=False):

        shape_series = []
        if not Test:
            prefix = 'train_sequences.csv_'
        else:
            prefix = 'test_sequences.csv_'
        for shape in shapes:
            file_path = os.path.join(self.shape_path, prefix + shape + '.csv')
            total_rows = pd.read_csv(file_path).shape[0]
            shape_data = pd.read_csv(file_path,  nrows=total_rows - 1)
            shape_series.append(shape_data)

        seq_num = shape_series[0].shape[0]
        completed_shape = np.empty((seq_num, len(shapes), shape_series[0].shape[1]))

        for i in range(len(shapes)):
            shape_samples = shape_series[i]
            for m in range(seq_num):
                completed_shape[m, i, :] = shape_samples.iloc[m].values
        completed_shape = np.nan_to_num(completed_shape)
        return completed_shape

class SSDataset_690(Dataset):

    def __init__(self, file_name, Test=False):
        shapes = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']
        sample_reader = SampleReader(file_name=file_name)
        self.completed_seqs, self.completed_labels = sample_reader.get_seq(Test=Test)

        self.completed_shape = sample_reader.get_shape(shapes=shapes, Test=Test)

    def __getitem__(self, item):
        return self.completed_seqs[item], self.completed_shape[item], self.completed_labels[item]

    def __len__(self):
        return self.completed_seqs.shape[0]