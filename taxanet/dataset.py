import torch
import numpy as np
from torch.utils.data import Dataset
import skbio


dna_alphabet = {b'A': 0, b'C': 1, b'G': 2, b'T': 3}


class DefaultMapper:

    def __init__(self):
        self.count = 0
        self.map = dict()

    def __call__(self, item):
        if item not in self.map:
            self.map[item] = self.count
            self.count += 1
        return self.map[item]

    def __len__(self):
        return self.count


class DictMapper:

    def __init__(self, dictionary, default=None):
        self.map = dictionary

        if default is None:
            def call(item): return self.map[item]
        else:
            def call(item): return self.map.get(item, default)
        self.call = call

    def __call__(self, item):
        return self.call(item)

    def __len__(self):
        return len(self.map)

    def keys(self):
        return self.map.keys()

    def items(self):
        return self.map.items()

    def values(self):
        return self.map.values()


def one_hot(seqs):
    """

    Parameters
    ----------
    seqs : np.array of bytes with shape (*, )

    Returns
    -------
    ohe : np.array of ints with shape (*, 5)
        where the positions indicate:

        - 0 : A
        - 1 : C
        - 2: G
        - 3: T
        - 4: other

    """
    idx = np.vectorize(dna_alphabet.get)(seqs, 4)
    ohe = torch.nn.functional.one_hot(torch.tensor(idx), 5)
    return ohe


class FastaCollection(Dataset):

    def __init__(self, fasta_list, read_length, mapper=None,
                 mapper_type='file',
                 ):
        """
        TODO could add a `mapper_type` object that adjusts how the mapper is
         applied to the given read, e.g., classify based on fasta id,
         or classify based on other classifier

        Parameters
        ----------
        fasta_list
        read_length
        mapper : dict or callable, optional (default=None)
            Must map any input to some int
        mapper_type : str
            Options:

            - 'file': maps entire file to the same value
            TODO
            - 'read': maps each entry based on the contents of the read
            - 'contig': maps each contig based on its id


        """
        self.read_length = read_length
        if mapper is None:
            class_map = DefaultMapper()
            self.class_map = class_map
        elif isinstance(mapper, dict):
            self.class_map = DictMapper(mapper)
        elif isinstance(mapper, callable):
            self.class_map = mapper
        else:
            raise ValueError('map_fn must be dict, callable, or None. Got '
                             '{}'.format(type(mapper)))

        map_dict = {'file': self.file_mapper,
                    }
        map_applier = map_dict[mapper_type]
        all_seqs = []
        all_labels = []

        self.fasta_list = fasta_list

        for file_ in self.fasta_list:
            for seq in skbio.io.read(file_, format='fasta'):
                all_seqs.append(seq.values)
                labels = np.full_like(seq.values, np.nan, dtype=np.float)
                map_applier(seq, read_length, self.class_map, labels, file_,
                            )
                all_labels.append(labels)

        self.seqs = np.hstack(all_seqs)
        self.labels = np.hstack(all_labels)
        self.is_valid = 1 - np.isnan(self.labels)
        self.sampleable_indices, = np.nonzero(self.is_valid)
        self.labels = self.labels.astype(np.int, copy=False)
        self.length = len(self.sampleable_indices)
        if self.length <= 0:
            raise ValueError('No contig is long enough to sample for '
                             'read_length={}'.format(read_length))

    def file_mapper(self, seq, read_length, mapper, labels, file_,
                    ):
        if len(seq) > read_length - 1:
            labels[:-read_length] = mapper(file_)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        valid_idx = self.sampleable_indices[idx]
        return one_hot(self.seqs[valid_idx:valid_idx + self.read_length]), \
            self.labels[valid_idx]
