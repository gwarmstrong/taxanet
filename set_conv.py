import torch
from torch import nn
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

default_alphabet = ['A', 'C', 'G', 'T']


class DNAFilterConstructor(BaseEstimator, TransformerMixin):

    def __init__(self, alphabet=None, dtype=None):
        if alphabet is None:
            alphabet = default_alphabet
        self.alphabet = alphabet
        self.one_hot_encoder = OneHotEncoder(categories=[self.alphabet],
                                             handle_unknown='ignore')
        self.kmer_length = None
        # higher lambda means penalize missing bases more
        self.lambda_ = 10
        self.hit_value = 1
        self.miss_value = None
        if dtype is None:
            self.dtype = np.float32

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : list or array of string
        y

        Returns
        -------

        """
        first_length = len(X[0])
        for i, kmer in enumerate(X):
            if len(kmer) != first_length:
                raise ValueError(f"All kmers must have same length. First "
                                 f"length is {first_length}. Got {len(kmer)}"
                                 f"for kmer {i}")
        self.kmer_length = first_length
        # this ensures that the miss value with be at least lamba less than
        #  the maximum activation score
        self.miss_value = -1 * (self.hit_value * self.kmer_length) - \
            self.lambda_
        # just need to fit it so sklearn doesnt get mad, even though
        # categories are set...
        self.one_hot_encoder.fit(np.array([list(X[0])]).transpose())
        return self

    def fit_transform(self, X, y=None, **fit_args):
        """

        Parameters
        ----------
        X : list or array of string
        y

        Returns
        -------

        """
        self.fit(X, y=y)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """

        Parameters
        ----------
        X : list or array of string
        y

        Returns
        -------

        """
        all_filters = [None for _ in range(len(X))]
        for i, filter_ in enumerate(X):
            all_filters[i] = self.one_hot_encoder.transform(
                [[letter] for letter in filter_]).toarray().transpose()
        all_filters = np.stack(all_filters, 0).astype(self.dtype)
        return all_filters * (-self.miss_value + 1) + self.miss_value


if __name__ == "__main__":
    # looks like: nn.Conv1d(4, num_channels, kernel_size=kmer_length)
    # so we're looking for 12 kmers of shape 9
    # bias is shape 12 (if bias)
    # weight is shape [12, 4, 9]
    demo_9_mers = [
        "ACGTGATCA",
        "GATTACTCA",
        "GACTAGTAA",
        "GACTAGTAA",
        "ACGTGATCT",
        "GATTACTCT",
        "GACTAGTAT",
        "GACTAGTAT",
        "ACGTGATCG",
        "GATTACTCG",
        "GACTAGTAG",
        "GACTAGTAG",
    ]
    dfc = DNAFilterConstructor()
    weights_9mers = dfc.fit_transform(demo_9_mers)
    conv = nn.Conv1d(4, 1, kernel_size=9, bias=False)
    # print(conv.bias.shape)
    # TODO -10 should be replace with more negative value... needs to be at
    #  most (k - 10) so its pretty well into negative territory all but one
    #  base matches
    demo_9mer_weights_variable = nn.Parameter(torch.tensor(weights_9mers),
                                              requires_grad=True)

    conv.weight = demo_9mer_weights_variable
    print(conv.weight)
