from unittest import mock
import torch
from torch import nn
from torch import optim
from taxanet.kraken_net import KrakenNet
from taxanet.set_conv import DNAStringOneHotEncoder, default_alphabet
import logomaker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import MDS

from taxanet.transformer import TransformerNetwork


def plot_logos(weight, bias=None, alphabet=None):
    if alphabet is None:
        alphabet = default_alphabet
    dfs = list()
    # create df for each conv
    for i, conv in enumerate(weight):
        df = pd.DataFrame(conv.t().detach().numpy(), columns=alphabet)
        if bias:
            df = df - bias
        dfs.append(df)

    # subtract bias, if it exists
    # create plot to view
    fig, axs = plt.subplots(len(dfs), 1,
                            figsize=(4, 5 * len(dfs)),
                            sharex=True,
                            )
    for ax, df in zip(axs, dfs):
        print(df)
        logomaker.Logo(df,
                       fade_below=0.5,
                       shade_below=0.5,
                       ax=ax,
                       )
    return fig, axs


def plot_internal_states(*args, scatterplot_kwargs):
    fig, axs = plt.subplots(len(args), 1,
                            figsize=(4, 5 * len(args)),
                            )
    for ax, X in zip(axs, args):
        mds = MDS(n_components=2)
        state = mds.fit_transform(X)
        sns.scatterplot(state[:, 0], state[:, 1], ax=ax,
                        **scatterplot_kwargs)
    print(len(args))


def train_net():
    kmer_length = 3
    channels = 8
    my_tree = \
        "1   |   1   |   no rank |\n" \
        "2   |   1   |   no rank |\n" \
        "3   |   1   |   no rank |\n" \
        "4   |   3   |   no rank |\n" \
        "5   |   3   |   no rank |\n" \
        "6   |   5   |   no rank |\n" \
        "7   |   6   |   no rank |\n" \
        "8   |   6   |   no rank |\n"
    database = {
        "AAT": 2,
        "ATT": 3,
        "ATA": 4,
        "TTT": 5,
        "TTC": 5,
        "TTG": 6,
        "TGC": 7,
        "GCA": 8,
    }
    test_data = [
        "AATTA",  # exp 1
        "AATTG",  # exp 6
        "GGGGG",  # exp 0
        "TTGCA",  # exp 6
        "TTTTT",  # exp 6
        "ATATT",  # exp 4
        "TTTTC",  # exp 6
        "GCAAA",  # exp 8
        "TGCCC",  # exp 7
        "AATCA",  # exp 2
        "GCATA",  # exp 3
    ]
    targets = torch.tensor([1, 6, 0, 6, 6, 4, 6, 8, 7, 2, 3])
    ohe = DNAStringOneHotEncoder()
    X = torch.tensor(ohe.fit_transform(test_data))
    # this mock lets me pass a string io in for the tree and have it be
    #  read as a file
    mocked_open_function = mock.mock_open(read_data=my_tree)
    with mock.patch("builtins.open", mocked_open_function):
        model = KrakenNet(kmer_length, channels, my_tree, train_lca=True,
                          train_rtl_sums=False)
    model.init_from_database(database, requires_grad=False)
    # for param in model.parameters():
    #     param.requires_grad = True
    model.weighted_lca_net.normalize = False

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            w_gain = nn.init.calculate_gain('tanh', m.weight)
            nn.init.xavier_uniform(m.weight.data, gain=w_gain)
            if m.bias:
                b_gain = nn.init.calculate_gain('tanh', m.bias)
                nn.init.xavier_uniform(m.bias.data, gain=b_gain)

    model.apply(weights_init)

    # model.init_from_database(database, requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(list(model.parameters()))
    n_epochs = 100000
    acc = 0
    # for i in range(n_epochs):
    i = 0
    loss = 100
    while not (acc == 1 and loss < 0.001):
        # zero the parameter gradients
        optimizer.zero_grad()
        print(f"epoch {i}")
        y = model(X)
        loss = loss_fn(y, targets)
        obs_classes = torch.argmax(y, 1)
        print(obs_classes, targets)
        acc = (obs_classes == targets).float().mean()
        print("accuracy", acc)
        print(f"loss: {loss}")
        loss.backward()
        optimizer.step()
        i += 1

    plot_logos(model.kmer_filter.weight, alphabet=ohe.alphabet)
    plt.show()


def train_transformer():
    test_data = [
        "AATTA",  # exp 1
        "AATTG",  # exp 6
        "GGGGG",  # exp 0
        "TTGCA",  # exp 6
        "TTTTT",  # exp 6
        "ATATT",  # exp 4
        "TTTTC",  # exp 6
        "GCAAA",  # exp 8
        "TGCCC",  # exp 7
        "AATCA",  # exp 2
        "GCATA",  # exp 3
    ]
    n_classes = 9
    delay = 1000
    targets = torch.tensor([1, 6, 0, 6, 6, 4, 6, 8, 7, 2, 3],
                           requires_grad=False)
    target_dist = torch.zeros(len(test_data), 9)
    target_dist[np.arange(len(test_data)), targets] = 1
    target_dist = target_dist.mean(dim=0)
    print("target dist", target_dist)

    ohe = DNAStringOneHotEncoder()
    X = torch.tensor(ohe.fit_transform(test_data))
    print("X shape:", X.shape)
    X = X.unsqueeze(2)
    print("Xu shape:", X.shape)

    model = TransformerNetwork(n_classes, kmer_length=4,
                               num_channels=10, nhead=5)
    loss_fn = nn.NLLLoss()
    # dist_loss_fn = nn.KLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # for i in range(n_epochs):
    i = 0
    i_since_1 = 0
    continue_ = True
    while continue_:
        # zero the parameter gradients
        optimizer.zero_grad()
        print(f"epoch {i}")
        y = model(X)
        # print("y", y)
        probs = y.mean(dim=0)
        print("probs", probs)
        loss = loss_fn(torch.log(y), targets)
        # loss = dist_loss_fn(torch.log(probs), target_dist)
        obs_classes = torch.argmax(y, 1)
        print(obs_classes, targets)
        acc = (obs_classes == targets).float().mean()
        print("accuracy", acc)
        print(f"loss: {loss}")
        if acc == 1:
            i_since_1 += 1
        else:
            i_since_1 = 0
        if i_since_1 > delay:
            break
        loss.backward()
        optimizer.step()
        i += 1

    print("weight shape", model.embedding.weight.shape)
    plot_logos(model.embedding.weight.squeeze(2), alphabet=ohe.alphabet)
    plt.show()
    model.eval()
    with torch.no_grad():
        X1 = model.embed(X)
        X2 = model.transformer_encoder(X1)
        hues = ['r' + str(i) for i in targets.numpy()]
        hue_order = list(sorted(set(hues)))
        print(hues)
        plot_internal_states(X1[0], X2[0],
                             scatterplot_kwargs={'hue': hues,
                                                 'hue_order': hue_order,
                                                 })
    plt.show()


if __name__ == "__main__":
    train_transformer()
