import unittest
from unittest import mock
import kraken_net
import builtins
import torch
import numpy.testing as npt
from kraken_net import (_read_nodes_dmp, _get_leaves_nodes, KrakenNet,
                        _preoder_traversal, _preoder_traversal_tuples,
                        _invert_tree, _postorder_traversal,
                        WeightedLCANet,
                        )
from set_conv import DNAStringOneHotEncoder


class KrakenNetTestCase(unittest.TestCase):

    def test_read_nodes_dmp(self):
        nodes = _read_nodes_dmp('../small_demo/taxonomy/nodes.dmp')
        exp_dict = {
            1: [3, 2, 7],
            2: [4, 5, 6],
        }
        for key in nodes:
            self.assertTrue(key in exp_dict)
            self.assertCountEqual(nodes[key], exp_dict[key])
        for key in exp_dict:
            self.assertTrue(key in nodes)

    def test_get_leaves_nodes(self):
        nodes = {
            1: [3, 2, 7],
            2: [4, 5, 6],
            6: [8],
        }
        leaves, nodes = _get_leaves_nodes(nodes)
        exp_leaves = [0, 3, 7, 4, 5, 8]
        exp_nodes = set(range(0, 9))
        self.assertCountEqual(leaves, exp_leaves)
        self.assertSetEqual(nodes, exp_nodes)

    def test_invert_tree(self):
        nodes = {
            1: [3, 2, 7],
            2: [4, 5, 6],
            6: [8],
        }
        inverted = _invert_tree(nodes)
        expected_inversion = {
            3: 1,
            2: 1,
            7: 1,
            4: 2,
            5: 2,
            6: 2,
            8: 6,
        }
        self.assertDictEqual(inverted, expected_inversion)

    def test_preorder_traversal(self):
        nodes = {
            1: [3, 2, 7],
            2: [4, 5, 6],
            6: [8],
        }
        exp_order = [1, 3, 2, 4, 5, 6, 8, 7]
        order = _preoder_traversal(nodes)
        self.assertListEqual(order, exp_order)

    def test_postorder_traversal(self):
        nodes = {
            1: [3, 2, 7],
            2: [4, 5, 6],
            6: [8],
        }
        exp_order = [3, 4, 5, 8, 6, 2, 7, 1]
        order = _postorder_traversal(nodes)
        self.assertListEqual(order, exp_order)

    def test_preorder_traversal_tuples(self):
        nodes = {
            1: [3, 2, 7],
            2: [4, 5, 6],
            6: [8],
        }
        exp_order = [(1, 3), (1, 2), (2, 4), (2, 5), (2, 6), (6, 8), (1, 7)]
        order = _preoder_traversal_tuples(nodes)
        self.assertListEqual(order, exp_order)

    def test_KrakenNet(self):
        N = 20
        read_length = 30
        kmer_length = 9
        channels = 5
        nodes_dmp = '../small_demo/taxonomy/nodes.dmp'
        self.X = torch.randn(N, 4, read_length)
        model = KrakenNet(kmer_length, channels, nodes_dmp)
        y = model(self.X)
        self.assertTupleEqual((N, model.n_nodes), y.shape)
        y.sum().backward()

    def test_KrakenNet_correctness(self):
        N = 7
        read_length = 5
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
            "AATTC",  # exp 1
            "AATTG",  # exp 6
            "GGGGG",  # exp 0
            "TTGCA",  # exp 6
            "TTTTT",  # exp 5
            "ATATT",  # exp 4
            "TTTTC",  # exp 5
        ]
        ohe = DNAStringOneHotEncoder()
        X = torch.tensor(ohe.fit_transform(test_data))
        mocked_open_function = mock.mock_open(read_data=my_tree)
        with mock.patch("builtins.open", mocked_open_function):
            model = KrakenNet(kmer_length, channels, my_tree)

        # weird, but I have to
        model.init_from_database(database, requires_grad=True)

        y = model(X)
        # self.assertTupleEqual((N, model.n_nodes), y.shape)
        obs_classes = torch.argmax(y, 1)
        exp_classes = [1, 6, 0, 6, 5, 4, 5]
        npt.assert_array_equal(obs_classes, exp_classes)

    def test_WeightedLCANet_simple(self):
        N = 20
        nodes_dmp = '../small_demo/taxonomy/nodes.dmp'
        tree = _read_nodes_dmp(nodes_dmp)

        # nodes includes 0, which is unclassified
        leaves, nodes = _get_leaves_nodes(tree)

        model = WeightedLCANet(tree, leaves, nodes)
        X = torch.randn(N, len(leaves), requires_grad=True)
        y = model(X)
        self.assertTupleEqual((N, len(nodes)), y.shape)
        y.sum().backward()

    def test_WeightedLCANet(self):
        N = 20
        tree = {
            1: [2, 3],
            2: [4, 5, 6],
            3: [7, 8],
        }
        # nodes includes 0, which is unclassified
        leaves, nodes = _get_leaves_nodes(tree)

        # override leaves to get the desired order
        leaves = [0, 4, 5, 6, 7, 8]
        model = WeightedLCANet(tree, leaves, nodes)
        X = torch.tensor([
            [0, 12, 12, 9, 10, 0],  # LCA is 2
            [0, 11, 11, 9, 10, 0],  # LCA is 2
            [0,  4,  4, 3, 12, 0],  # LCA is 7
            [0,  4,  4, 3,  9, 9],  # LCA is 3
            [9,  0,  0, 0,  0, 0],  # LCA is 0
            [1,  1,  0, 0,  0, 0],  # LCA is 4
            [1,  5,  5, 5,  0, 0],  # LCA is 2
        ],
            requires_grad=True,
            dtype=torch.float32,
        )
        y = model(X)
        # printing y here can be useful for visualizing what the outputs
        # look like
        # print(y)
        obs_classes = torch.argmax(y, 1)
        exp_classes = [2, 2, 7, 3, 0, 4, 2]
        npt.assert_array_equal(obs_classes, exp_classes)
        self.assertTupleEqual((len(X), len(nodes)), y.shape)
        y.sum().backward()

    def test_WeightedLCANet_inexact(self):
        tree = {
            1: [2, 3],
            2: [4, 5, 6],
            3: [7, 8],
        }
        # nodes includes 0, which is unclassified
        leaves, nodes = _get_leaves_nodes(tree)

        # override leaves to get the desired order
        leaves = [0, 4, 5, 6, 7, 8]
        model = WeightedLCANet(tree, leaves, nodes)
        X = torch.tensor([
            [0, 12, 11.99, 9, 10, 0],  # LCA is 2
            [0, 11, 11, 9, 10, 0],  # LCA is 2
            [0,  4,  4, 3, 12, 0],  # LCA is 7
            [0,  4,  4, 3,  9, 8.1],  # LCA is 3
            [9,  0,  0, 0,  0, 0],  # LCA is 0
            [2,  1,  0, 0,  0, 0],  # LCA is 4
            [1,  5.05,  5, 5,  0, 0],  # LCA is 2
        ],
            requires_grad=True,
            dtype=torch.float32,
        )
        y = model(X)
        # printing y here can be useful for visualizing what the outputs
        # look like
        # print(y)
        obs_classes = torch.argmax(y, 1)
        exp_classes = [2, 2, 7, 3, 0, 4, 2]
        npt.assert_array_equal(obs_classes, exp_classes)
        self.assertTupleEqual((len(X), len(nodes)), y.shape)
        y.sum().backward()

if __name__ == '__main__':
    unittest.main()
