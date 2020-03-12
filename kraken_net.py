import torch
from torch import nn
import warnings
import logging
import sys
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


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
        self.max_value = 1 / (1 + self.epsilon)
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

        sum_leaves = maxes_rep.sum(dim=1)
        # Xmax_excl_unc[:, 1:], _ = X[:, 1:].max(dim=1)
        Xmax_excl_unc[:, 1:] = maxes_rep
        logging.debug(f"leaves: {self.leaves}")
        logging.debug(f"X-meu:\n{Xmax_excl_unc}")
        X = X - Xmax_excl_unc + self.max_value
        # now relu so only thing that score within max_value of the max are
        # included
        logging.debug(f"X:\n{X}")

        X = self.relu(X)
        result = torch.zeros(X.shape[0], len(self.nodes))
        for i, leaf in enumerate(self.leaves):
            if leaf != 0:
                # TODO think carefully about whether sum_leaves shoudl be
                #  multiplied.
                #  It was added to combat the problem of no nodes being
                #  assigned
                result[:, leaf] = X[:, i] * (sum_leaves / (sum_leaves +
                                                           self.epsilon))
            else:
                # TODO for now this assumes that epsilon * X[:, 0] is less than
                #  max_value (around 1). But greater than 0 (assumes X[:,
                #  0] non-negative)
                result[:, leaf] = self.epsilon * (X[:, i] + 1)

        logging.debug(f"nodes {self.nodes}")
        logging.debug(f"results(pre traversal):\n{result}")

        for node in self.postorder_traveral:
            if node not in self.leaves:
                for child in self.parent_to_children[node]:
                    # multiply by (1 - epsilon) so a parent is only larger
                    # than child if it has multiple descendents
                    result[:, node] += (1 - self.epsilon) * result[:, child]
                # result[:, node] = result[:, node] / (result[:, node] +
                #                                      self.epsilon)
        # result is now a (N, n_nodes) tensor
        logging.debug(f"results(post traversal):\n{result}")
        return result


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
        # TODO bias?
        self.kmer_filter = nn.Conv1d(4, num_channels, kernel_size=kmer_length)
        self.linear_layer = nn.Linear(num_channels, self.n_nodes)
        # TODO activation?
        self.weighted_lca_net = WeightedLCANet(self.tree, self.leaves,
                                               self.nodes)
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
        # returns shape (N, read_length - kmer_length + 1, n_nodes)
        taxa_affinities = self.linear_layer(feature_map)
        # todo here is where the activation should go

        root_to_node_sums = {
            0: taxa_affinities[:, :, 0],
            1: taxa_affinities[:, :, 1],
        }
        # TODO ASSUMES root is 1
        # in a preorder traversal, the parent of a node will be calculated
        # before it, so it is safe to pass on to their children
        for node in _preoder_traversal(self.tree):
            # pass on no
            if node in self.tree:
                for child in self.tree[node]:
                    root_to_node_sums[child] = root_to_node_sums[node] + \
                        taxa_affinities[:, :, child]
        # TODO assumes that all nodes are present in tree...
        # should be a (N, n_leaves, length - kmer_length + 1) tensor
        # Note: 0 (unclassified is included in this)
        rtl_sums = torch.stack([root_to_node_sums[i] for i in
                               self.leaves],
                               dim=1,
                               )
        rtl_sums = rtl_sums.sum(dim=2)
        # todo sum across the kmers. If padding is added, mask the sum...
        # input to LCA will be (N, n_leaves)
        # output from LCA wil be (N, n_nodes)

        lca = self.weighted_lca_net(rtl_sums - rtl_sums.max())
        return lca
