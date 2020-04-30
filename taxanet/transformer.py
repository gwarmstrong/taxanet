from torch import nn
from taxanet.densenet import DenseBlock1D


class TransformerNetwork(nn.Module):

    def __init__(self, n_classes, kmer_length, num_channels, num_layers=2,
                 nhead=6):
        super().__init__()
        self.embedding = nn.Conv2d(4, num_channels,
                                   kernel_size=(1, kmer_length), bias=False)
        self.embedding_length = num_channels
        self.nhead = nhead
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_length, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=num_layers)
        self.linear = nn.Linear(self.embedding_length, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def embed(self, X):
        X = self.embedding(X)
        X = self.avg_pool(X).squeeze(2).squeeze(2).unsqueeze(0)
        return X

    def forward(self, X):
        # input shape: (N_reads, 4, 1, len_reads)
        X = self.embedding(X)
        # input shape: (N_reads, num_channels, 1, pool_down_size)
        X = self.avg_pool(X).squeeze(2).squeeze(2).unsqueeze(0)
        # input shape: (1, N_reads, num_channels)
        X = self.transformer_encoder(X)
        X = self.linear(X).squeeze(0)
        X = self.softmax(X)
        return X


class TransformerDenseNetwork(nn.Module):

    def __init__(self, n_classes, kmer_length, num_layers=2,
                 nhead=6, ndense_layers=6, growth_rate=32, conv_bn_size=4):
        super().__init__()
        self.embedding = DenseBlock1D(ndense_layers, 4,
                                      kernel_size=kmer_length,
                                      bn_size=conv_bn_size,
                                      growth_rate=growth_rate
                                      )
        self.embedding_length = 4 + growth_rate * num_layers
        self.nhead = nhead
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_length, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=num_layers)
        self.linear = nn.Linear(self.embedding_length, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def embed(self, X):
        X = self.embedding(X)
        X = self.avg_pool(X).squeeze(2).squeeze(2).unsqueeze(0)
        return X

    def forward(self, X):
        # input shape: (N_reads, 4, 1, len_reads)
        X = self.embedding(X)
        print(f"X after embed: {X.shape}")
        # input shape: (N_reads, num_channels, 1, pool_down_size)
        X = self.avg_pool(X).squeeze(2).squeeze(2).unsqueeze(0)
        print(f"X after pool: {X.shape}")
        # input shape: (1, N_reads, num_channels)
        X = self.transformer_encoder(X)
        X = self.linear(X).squeeze(0)
        X = self.softmax(X)
        return X
