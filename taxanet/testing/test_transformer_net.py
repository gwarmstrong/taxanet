from unittest import TestCase
import torch
from taxanet.transformer import TransformerNetwork, TransformerDenseNetwork
from taxanet.densenet import DenseBlock1D


class TransformerTestCase(TestCase):

    def setUp(self):
        # some random demo 10 reads of length 150
        self.X = torch.randn(10, 4, 1, 150)

    def test_transformer_runs(self):
        model = TransformerNetwork(n_classes=3,
                                   kmer_length=5, num_channels=10, nhead=5,
                                   )
        y = model(self.X)

        self.assertTupleEqual((10, 3), y.shape)

    def test_transformer_dense_runs(self):
        model = TransformerDenseNetwork(n_classes=3, num_layers=6,
                                        kmer_length=3, growth_rate=10, nhead=8,
                                        )
        y = model(self.X)

        self.assertTupleEqual((10, 3), y.shape)

    def test_denseblock_runs(self):
        model = DenseBlock1D(num_layers=2, num_input_features=4)
        print(model)
        y = model(self.X)
        print(f"y size: {y.shape}")

