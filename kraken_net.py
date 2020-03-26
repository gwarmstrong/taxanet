import math
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import init
from itertools import product
import warnings
import logging
import sys
from set_conv import DNAFilterConstructor
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class ReLUMiddleLinear(nn.Module):
    """
    Does like a linear layer, but only takes the positive terms in the
    matrix multiplication (before summing)

    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(ReLUMiddleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.relu = nn.ReLU()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        # return F.linear(input, self.weight, self.bias)
        # output = input.matmul(self.weight.t())
        output = torch.einsum('bij,jk->bikj', input, self.weight.t())
        output = self.relu(output)
        output = output.sum(dim=3)
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def _read_nodes_dmp(fp):
    # TODO I'm sure Qiyun has written something like this
    """

    Parameters
    ----------
    fp

    Returns
    -------

    """
    with open(fp) as fo:
        lines = fo.readlines()
    nodes = dict()
    all = set()
    for line in lines:
        parts = line.split('|')
        child = int(parts[0].strip())
        parent = int(parts[1].strip())
        if parent == 0 or child == 0:
            raise ValueError("nodes file cannot contain any nodes labeled "
                             "as 0")
        all.add(child)
        all.add(parent)
        if parent == child:
            continue
        if parent in nodes:
            nodes[parent].append(child)
        else:
            nodes[parent] = [child]
    n_nodes = len(all)
    expected_nodes = set(range(1, n_nodes + 1))
    if not all == expected_nodes:
        raise ValueError("nodes must be numerated 1...n_nodes with all "
                         "intermediates.")
    return nodes


def _invert_tree(nodes):
    inverted = dict()
    for parent in nodes:
        for child in nodes[parent]:
            # should not happen, but could if tree is malformed..
            if child in inverted:
                raise ValueError(f"Node {child} already added.")
            inverted[child] = parent
    return inverted


def _get_leaves_nodes(nodes):
    """
    Assumes that nodes was loaded with _read_nodes_dmp. Thus no nodes can be 0
    as this is reserved for "unclassified"

    Parameters
    ----------
    nodes : dict

    Returns
    -------

    """
    parents = set()
    children = set()
    for node in nodes:
        parents.add(node)
        children.update(nodes[node])
    children.add(0)
    leaves = list(sorted(children - parents))
    nodes = children | parents
    return leaves, nodes


def _preoder_traversal(nodes, root=1, order=None):
    """
    TODO assumes that 1 is the root of the tree
    Parameters
    ----------
    nodes

    Returns
    -------

    """
    if order is None:
        order = []
    order.append(root)
    if root in nodes:
        for child in nodes[root]:
            _preoder_traversal(nodes, root=child, order=order)
    return order


def _preoder_traversal_tuples(nodes, root=1, order=None):
    """
    TODO assumes that 1 is the root of the tree
    Parameters
    ----------
    nodes

    Returns
    -------

    """
    if order is None:
        order = []
    if root in nodes:
        for child in nodes[root]:
            order.append((root, child))
            _preoder_traversal_tuples(nodes, root=child, order=order)
    return order


def _postorder_traversal(nodes, root=1, order=None):
    if order is None:
        order = []
    if root in nodes:
        for child in nodes[root]:
            _postorder_traversal(nodes, root=child, order=order)
    order.append(root)
    return order


def _get_nodes_to_leaves(nodes, root=1, mapping=None):
    if mapping is None:
        mapping = {0: [0]}
    mapping[root] = []
    if root in nodes:
        for child in nodes[root]:
            _get_nodes_to_leaves(nodes, root=child, mapping=mapping)
            mapping[root].extend(mapping[child])
    else:
        # it is a leaf, so give it itself as a "descending" leaf
        mapping[root].append(root)

    return mapping


def _get_nodes_to_all_descendents(nodes, root=1, mapping=None):
    if mapping is None:
        mapping = {0: {0}}
    mapping[root] = set()
    if root in nodes:
        for child in nodes[root]:
            _get_nodes_to_all_descendents(nodes, root=child, mapping=mapping)
            mapping[root].update(mapping[child])
            mapping[root].add(child)
    else:
        mapping[root].add(root)
    return mapping


def _get_nodes_to_all_ancestors(nodes, root=1, ancestors=None):
    if ancestors is None:
        ancestors = {0: set()}
    if root not in ancestors:
        ancestors[root] = set()
    if root in nodes:
        for child in nodes[root]:
            if child not in ancestors:
                ancestors[child] = set()
            ancestors[child].update(ancestors[root])
            ancestors[child].add(root)
            _get_nodes_to_all_ancestors(nodes, root=child, ancestors=ancestors)
    return ancestors


def tanh_onto_0_to_1(x):
    x = torch.mul(0.5, torch.add(x, 1))
    return x


class WeightedLCANet(nn.Module):

    def __init__(self, parent_to_children, leaves=None, nodes=None):
        super().__init__()
        self.parent_to_children = parent_to_children
        self.leaves = leaves
        self.nodes = nodes
        if leaves is None or nodes is None:
            if leaves is not None or nodes is not None:
                raise UserWarning("Received None for either leaves or nodes."
                                  " Recalculating both.")
            self.leaves, self.nodes = _get_leaves_nodes(self.tree)
        self.epsilon = 1e-5
        self.max_value = 1 # / (1 + self.epsilon)
        self.postorder_traveral = _postorder_traversal(self.parent_to_children)
        self.relu = nn.ReLU()
        if not isinstance(self.leaves, list) and self.leaves:
            raise ValueError(f"Expected non-empty list for leaves. "
                             f"Got {self.leaves}.")

    def forward(self, X):
        # todo how to handle 0...
        # assumes X has shape (N, n_leaves)
        # X should also be ordered in the same order as leaves
        if X.ndim != 2:
            raise ValueError(f"X expected to have 2 dimensions. Got "
                             f"{X.ndim}.")
        elif X.shape[1] != len(self.leaves):
            raise ValueError(f"X expected to be of shape (N, "
                             f"{len(self.leaves)}. Got (N, {X.shape[1]})")

        # normalize the matrix so the max count is ~1 (in max_value)
        # X.max() is not the
        # Xmax (by sample) excluding unclassified samples
        Xmax_excl_unc = torch.zeros_like(X)
        # assumes unclassified is in position 0
        row_maxes, _ = X[:, 1:].max(dim=1)
        maxes_rep = row_maxes.repeat_interleave(X.shape[1] - 1)\
            .view(X.shape[0], X.shape[1] - 1)

        # sum_leaves lets us know if anything is non-zero, or if all are zero
        sum_leaves = maxes_rep.sum(dim=1)
        # print("sum_leaves", sum_leaves)
        # Xmax_excl_unc[:, 1:], _ = X[:, 1:].max(dim=1)
        Xmax_excl_unc[:, 1:] = maxes_rep
        logging.debug(f"leaves: {self.leaves}")
        logging.debug(f"X-meu:\n{Xmax_excl_unc}")
        X = X - Xmax_excl_unc + self.max_value
        # now relu so only thing that score within max_value of the max are
        # included
        # logging.debug(f"X:\n{X}")

        X = self.relu(X)
        lca_sums = torch.zeros(X.shape[0], len(self.nodes))
        # lca_normalized = torch.zeros(X.shape[0], len(self.nodes))
        for i, leaf in enumerate(self.leaves):
            if leaf != 0:
                # TODO think carefully about whether sum_leaves shoudl be
                #  multiplied.
                #  It was added to combat the problem of no nodes being
                #  assigned
                # sum_leaves can vary in scale, so I should pay attention
                #  to this in theoretical calculations
                lca_sums[:, leaf] = X[:, i] * (sum_leaves / (sum_leaves +
                                                           self.epsilon))
            else:
                # TODO for now this assumes that epsilon * X[:, 0] is less than
                #  max_value (around 1). But greater than 0 (assumes X[:,
                #  0] non-negative)
                lca_sums[:, leaf] = self.epsilon * (X[:, i] + 1)

        # logging.debug(f"nodes {self.nodes}")
        # logging.debug(f"results(pre traversal):\n{lca_sums}")

        for node in self.postorder_traveral:
            if node not in self.leaves:
                for child in self.parent_to_children[node]:
                    # multiply by (1 - epsilon) so a parent is only larger
                    # than child if it has multiple descendents
                    lca_sums[:, node] += lca_sums[:, child]
                # norm_factor will make the sum greater than either
                # children, if multiple children are non-zero, and will make
                # it less than child if one child, and 0 if 0. Approximately
                # 1 in first two cases.
                norm_factor = (1 - self.epsilon)
                lca_sums[:, node] *= norm_factor
        # result is now a (N, n_nodes) tensor
        # logging.debug(f"results(post traversal):\n{lca_sums}")
        return lca_sums


class KrakenNet(nn.Module):

    def __init__(self, kmer_length, num_channels, nodes_dmp):
        super().__init__()
        self.tree = _read_nodes_dmp(nodes_dmp)
        self.children_to_parent = _invert_tree(self.tree)
        # nodes includes 0, which is unclassified
        self.leaves, self.nodes = _get_leaves_nodes(self.tree)
        self.n_leaves = len(self.leaves)
        self.n_nodes = len(self.nodes)
        self.num_channels = num_channels
        self.kmer_length = kmer_length
        # I think this is better with bias = False for now?
        self.kmer_filter = nn.Conv1d(4, num_channels,
                                     kernel_size=kmer_length, bias=False)
        self.linear_layer = nn.Linear(num_channels, self.n_nodes, bias=True)
        # switched to this layer because I need the positive parts of the
        # matrix mutliplication
        # https://stackoverflow.com/questions/47974959/numpy-matrix-
        #  multiplication-to-return-ndarray-not-sum
        # self.linear_layer = ReLUMiddleLinear(num_channels, self.n_nodes,
        #                                      bias=True)
        self.root_to_leaf_sums = MatrixRootToLeafSums(self.tree, self.leaves,
                                                      self.n_nodes
                                                      )
        # LCA net has no parameters!
        self.weighted_lca_net = MatrixLCANet(self.tree, self.leaves,
                                             self.nodes,
                                             alpha=0.01,
                                             )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.tanh_onto_0_to_1 = tanh_onto_0_to_1
        self.database = None
        self.database_filters = None
        self.filter_constructor = None

        # 1/2 tanh + 1/2

    def forward(self, X):
        """

        Parameters
        ----------
        X : torch.Tensor of shape (N, 4, read_length)
        TODO figure out how padding should work

        Returns
        -------

        """

        # returns shape (N, self.num_channels, read_length - kmer_length + 1)
        # then needs to be reshaped to (N, L, nc) for linear layer...
        feature_map = self.kmer_filter(X).permute(0, 2, 1).contiguous()
        # returns shape (N, read_length - kmer_length + 1, n_channels)
        feature_map = self.tanh(feature_map)
        feature_map = self.tanh_onto_0_to_1(feature_map)
        taxa_affinities = self.linear_layer(feature_map)
        # returns shape (N, n_nodes, read_length - kmer_length + 1)
        taxa_affinities = self.tanh(taxa_affinities)
        taxa_affinities = self.tanh_onto_0_to_1(taxa_affinities)

        # TODO wait why did I not need this? bias term?
        # basically, if any node is 1, this will be 0, and if no nodes are
        # 1, this will be 1 (in the near perfect case)
        # taxa_affinities[:, :, 0] = torch.add(
        #     -taxa_affinities[:, :, 1:].max(dim=2)[0], 1.)

        # print("taxaff\n", taxa_affinities)
        taxa_affinities = taxa_affinities.sum(dim=1)
        rtl_sums = self.root_to_leaf_sums(taxa_affinities)

        # input to LCA will be (N, n_leaves)
        # output from LCA wil be (N, n_nodes)
        lca = self.weighted_lca_net(rtl_sums)
        return lca

    def init_from_database(self, database, requires_grad=True):
        """

        Parameters
        ----------
        database : dict of str to int
            where the str is a kmer and int is node in the tree
        requires_grad : bool
            whether the filter params require a gradient


        Returns
        -------

        """
        # TODO this should be changed to a less than, so some channels can
        #  be left free to be learned (i.e., len(database) < self.num_channels
        if len(database) != self.num_channels:
            raise ValueError("Database must be the same length as number of "
                             "channels in the network.")
        # TODO rewrite so we're not looping over these a bunch of times...
        if not all(node in self.nodes for node in database.values()):
            # TODO improve this error message
            raise ValueError("Database must only contain nodes in the tree.")
        if not all(len(kmer) == self.kmer_length for kmer in database.keys()):
            # TODO improve this error message
            raise ValueError(f"Database must only contain kmers of length "
                             f"{self.kmer_length}")

        self.database = OrderedDict(database)
        self.database_filters = list(self.database.keys())

        self.filter_constructor = DNAFilterConstructor()
        filter_weights = self.filter_constructor.fit_transform(
            self.database_filters)
        # print(conv.bias.shape)
        # TODO -10 should be replace with more negative value... needs to be at
        #  most (k - 10) so its pretty well into negative territory all but one
        #  base matches
        filter_weights_param = nn.Parameter(torch.tensor(filter_weights),
                                            requires_grad=requires_grad)

        self.kmer_filter.weight = filter_weights_param

        # todo this -10 should be more flexible, but is to make sure that
        #  everything is off by default,
        kmer_map_bias = nn.Parameter(torch.tensor(
            np.repeat(-10, self.n_nodes).astype(np.float32)),
            requires_grad=requires_grad
        )

        # TODO initialize linear layer (n_nodes x n_channels)
        kmer_map = np.zeros((self.n_nodes, self.num_channels),
                            dtype=np.float32)
        nonzero_positions_x = []
        nonzero_positions_y = []
        for channel, filter_ in enumerate(self.database_filters):
            target_node = self.database[filter_]
            nonzero_positions_x.append(target_node)
            nonzero_positions_y.append(channel)

        # print("db_filt", self.database_filters)
        # print("positions", nonzero_positions_x, nonzero_positions_y)
        kmer_map[nonzero_positions_x, nonzero_positions_y] = 20.
        # print("kmer_map 5", kmer_map[5, :])
        # print("kmer_map\n", kmer_map)
        # print("kmer_map_bias\n", kmer_map_bias)
        kmer_map_param = nn.Parameter(torch.tensor(kmer_map),
                                      requires_grad=requires_grad,
                                      )

        self.linear_layer.bias = kmer_map_bias
        self.linear_layer.weight = kmer_map_param


class RootToLeafSums(nn.Module):

    def __init__(self, tree, leaves, n_nodes):
        self.tree = tree
        self.leaves = leaves
        self.n_nodes = n_nodes
        super().__init__()

    def forward(self, taxa_affinities):
        # expects shape (N_samples, n_nodes)
        root_to_node_sums = {
            0: taxa_affinities[:, 0],
            1: taxa_affinities[:, 1],
        }
        # TODO ASSUMES root is 1
        # in a preorder traversal, the parent of a node will be calculated
        # before it, so it is safe to pass on to their children
        for node in _preoder_traversal(self.tree):
            # pass on no
            if node in self.tree:
                for child in self.tree[node]:
                    root_to_node_sums[child] = root_to_node_sums[node] + \
                                               taxa_affinities[:, child]
        # TODO assumes that all nodes are present in tree...
        # should be a (N, n_leaves, length - kmer_length + 1) tensor
        # Note: 0 (unclassified is included in this)
        rtl_sums = torch.stack([root_to_node_sums[i] for i in
                                self.leaves],
                               dim=1,
                               )
        # rtl_sums = rtl_sums.sum(dim=2)
        # todo sum across the kmers. If padding is added, mask the sum...
        #  actually may not need to mask the sum? things will just be unmapped
        return rtl_sums


class MatrixRootToLeafSums(nn.Module):

    def __init__(self, tree, leaves, n_nodes):
        super().__init__()
        self.tree = tree
        self.leaves = {leaf: i for i, leaf in enumerate(leaves)}
        self.n_nodes = n_nodes
        self.linear = nn.Linear(self.n_nodes, len(leaves), bias=False)
        self._init_weights(self.linear)

    def forward(self, X):
        # expects shape (N_samples, n_nodes)
        # returns shape (N_samples, n_leaves)
        return self.linear(X)

    def _init_weights(self, layer):
        layer.weight = nn.Parameter(torch.zeros_like(layer.weight),
                                    requires_grad=False,
                                    )
        node_leaf_mapping = _get_nodes_to_leaves(self.tree)
        for node, tips in node_leaf_mapping.items():
            for leaf in tips:
                layer.weight[self.leaves[leaf], node] = 1.


class MatrixLCANet(nn.Module):

    def __init__(self, parent_to_children, leaves=None, nodes=None,
                 alpha=0.05,
                 ):
        super().__init__()
        self.tree = parent_to_children
        if leaves is None or nodes is None:
            if leaves is not None or nodes is not None:
                raise UserWarning("Received None for either leaves or nodes."
                                  " Recalculating both.")
            leaves, nodes = _get_leaves_nodes(self.tree)
        self.leaves = {leaf: i for i, leaf in enumerate(leaves)}
        self.n_leaves = len(self.leaves)
        self.n_nodes = len(nodes)
        self.nodes = nodes
        self.leaves_to_ancestors = nn.Linear(self.n_leaves, self.n_nodes,
                                             bias=True,
                                             )
        self.children_to_parents = nn.Linear(self.n_nodes, self.n_nodes,
                                             bias=True,
                                             )
        self.nodes_to_ancestors = nn.Linear(self.n_nodes, self.n_nodes,
                                            bias=True,
                                            )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.tanh_onto_0_to_1 = tanh_onto_0_to_1
        self.max_value = 1
        # alpha controls how permissive the max thresholding is
        self.alpha = alpha
        self._init_weights()

    def forward(self, X):
        # expects shape (N_samples, n_nodes)
        # returns shape (N_samples, n_leaves)
        # TODO turn this into a function
        #################################################
        # the scaling max onto 1 and rest onto 0 function
        #################################################
        print("leaves", self.leaves)
        print("input to LCA", X)
        if X.ndim != 2:
            raise ValueError(f"X expected to have 2 dimensions. Got "
                             f"{X.ndim}.")
        elif X.shape[1] != len(self.leaves):
            raise ValueError(f"X expected to be of shape (N, "
                             f"{len(self.leaves)}. Got (N, {X.shape[1]})")

        #######
        # try to get weights on to 0 and 1 based on max
        #######
        row_maxes, _ = X[:, 1:].max(dim=1)
        normalization = ((1 - self.alpha) * row_maxes).unsqueeze(1)
        print("norm terms", X)
        #         # 5 below should probably be a parameter to control where on the
        # tanh curve the max value ends up
        X = 100 * ((X - normalization) * 5 * (1 / self.alpha) - 1)
        print("mid-norm", X)
        X = self.tanh(X)
        X = self.tanh_onto_0_to_1(X)
        # if all reads are unclassified, we will get (tanh(0) = 0.5),
        #  so move them down and mulitply by 2 so max is still 1 and 0.5
        #  gets moved to 0
        # X = self.tanh_onto_0_to_1(self.tanh(20 * (X - 0.75)))
        print("post-norm", X)
        #######
        # end
        #######

        X[:, 0] = 0

        # map maximal-root-to-leaf-sum nodes to activate ancestors
        X = self.tanh_onto_0_to_1(self.tanh(self.leaves_to_ancestors(X)))
        print("child to ancestor map", X)
        # activate any node with two activated children
        X = self.tanh_onto_0_to_1(self.tanh(self.children_to_parents(X)))
        print("has multiple activated children", X)
        # TODO if it has an ancestor on, turn it off (since there is a
        #  higher lca)
        X = self.tanh_onto_0_to_1(self.tanh(self.nodes_to_ancestors(X)))
        print("lca predictions", X)
        return X

    def _init_weights(self):
        self.leaves_to_ancestors.weight = nn.Parameter(
            torch.zeros(self.n_nodes, self.n_leaves),
            requires_grad=False,
        )
        self.leaves_to_ancestors.bias = nn.Parameter(
            torch.ones(self.n_nodes),
            requires_grad=False,
        )
        self.children_to_parents.weight = nn.Parameter(
            torch.zeros(self.n_nodes, self.n_nodes),
            requires_grad=False,
        )
        self.children_to_parents.bias = nn.Parameter(
            torch.ones(self.n_nodes),
            requires_grad=False,
        )
        self.nodes_to_ancestors.weight = nn.Parameter(
            torch.zeros(self.n_nodes, self.n_nodes),
            requires_grad=False,
        )
        self.nodes_to_ancestors.bias = nn.Parameter(
            torch.ones(self.n_nodes),
            requires_grad=False,
        )
        # a parent can be activated if any one of its children is on
        self.leaves_to_ancestors.bias *= -10.
        # a node needs two children on to be activated
        self.children_to_parents.bias *= -20.
        # a node has a tendency to be off, can be activated by itself,
        # but turned off by any ancestor
        self.nodes_to_ancestors.bias *= -10.

        node_leaf_mapping = _get_nodes_to_leaves(self.tree)
        print("node to descending leaves mapping: ", node_leaf_mapping)
        for node, tips in node_leaf_mapping.items():
            for leaf in tips:
                # from leaf onto node
                # set bias from leaf to ancestor so it needs multiple leaves
                # on to turn on
                self.leaves_to_ancestors.weight[node, self.leaves[leaf]] = 15.
                # set leaf to leaf bias so that a single leaf can activate leaf
                self.leaves_to_ancestors.bias[leaf] = -10.

        node_child_mapping = self.tree
        for node, children in node_child_mapping.items():
            for child in children:
                # from child onto node
                self.children_to_parents.weight[node, child] = 15.

        leaves = self.leaves.keys()
        for leaf1, leaf2 in product(leaves, leaves):
            # from child onto node
            if leaf1 == leaf2:
                # turn leaf on if it is on
                self.children_to_parents.weight[leaf1, leaf1] = 15.
                # each leaf should meet this condition once, so give it bias
                #  here
                self.children_to_parents.bias[leaf1] = -5.
            # else:
            #     # turn leaf off if other leaves are on
            #     self.children_to_parents.weight[leaf1, leaf2] = -15.

        node_ancestor_mapping = _get_nodes_to_all_ancestors(self.tree)
        # a node has a tendency to be off (-10), can be activated by itself
        # (20), but turned off by any ancestor (-30)
        for node in node_ancestor_mapping:
            if node == 0:
                # has a tendency to be unclassified
                self.nodes_to_ancestors.bias[0] = 10.
                # but is not unclassified if anything is classified
                self.nodes_to_ancestors.weight[0, 1:] = -20.
            else:
                self.nodes_to_ancestors.weight[node, node] = 20.
            for ancestor in node_ancestor_mapping[node]:
                self.nodes_to_ancestors.weight[node, ancestor] = -30.

        # 0 has a tendency to be on, but is off if any nodes are on
        self.children_to_parents.bias[0] = 5
        for node in self.nodes:
            # from child onto node
            self.children_to_parents.weight[node, 0] = -15.


