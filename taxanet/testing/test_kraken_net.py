import unittest
from unittest import mock
import torch
import numpy.testing as npt
from taxanet.kraken_net import (_read_nodes_dmp, _get_leaves_nodes, KrakenNet,
                                _preoder_traversal, _preoder_traversal_tuples,
                                _invert_tree, _postorder_traversal,
                                _get_nodes_to_leaves, _get_nodes_to_all_descendents,
                                _get_nodes_to_all_ancestors,
                                WeightedLCANet, RootToLeafSums,
                                MatrixRootToLeafSums, MatrixLCANet,
                                )
from taxanet.set_conv import DNAStringOneHotEncoder


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

    def test_get_nodes_to_leaves(self):
        nodes = {
            1: [3, 2, 7],
            2: [4, 5, 6],
            6: [8],
        }
        exp_dict = {0: [0],
                    1: [3, 4, 5, 8, 7],
                    2: [4, 5, 8],
                    3: [3],
                    4: [4],
                    5: [5],
                    6: [8],
                    7: [7],
                    8: [8],
                    }
        nodes = _get_nodes_to_leaves(nodes)
        for key in nodes:
            self.assertTrue(key in exp_dict)
            self.assertCountEqual(nodes[key], exp_dict[key])
        for key in exp_dict:
            self.assertTrue(key in nodes)

    def test_get_nodes_to_descendents(self):
        nodes = {
            1: [3, 2, 7],
            2: [4, 5, 6],
            6: [8],
        }
        exp_dict = {0: [0],
                    1: [2, 3, 4, 5, 6, 8, 7],
                    2: [4, 5, 6, 8],
                    3: [3],
                    4: [4],
                    5: [5],
                    6: [8],
                    7: [7],
                    8: [8],
                    }
        nodes = _get_nodes_to_all_descendents(nodes)
        for key in nodes:
            self.assertTrue(key in exp_dict)
            self.assertCountEqual(nodes[key], exp_dict[key])
        for key in exp_dict:
            self.assertTrue(key in nodes)

    def test_get_nodes_to_ancestors(self):
        nodes = {
            1: [3, 2, 7],
            2: [4, 5, 6],
            6: [8],
        }
        exp_dict = {0: [],
                    1: [],
                    2: [1],
                    3: [1],
                    4: [1, 2],
                    5: [1, 2],
                    6: [1, 2],
                    7: [1],
                    8: [1, 2, 6],
                    }
        nodes = _get_nodes_to_all_ancestors(nodes)
        for key in nodes:
            self.assertTrue(key in exp_dict)
            self.assertCountEqual(nodes[key], exp_dict[key],
                                  msg=f"node: {key}, all obs: {nodes}")
        for key in exp_dict:
            self.assertTrue(key in nodes)

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
        ohe = DNAStringOneHotEncoder()
        X = torch.tensor(ohe.fit_transform(test_data))
        # this mock lets me pass a string io in for the tree and have it be
        #  read as a file
        mocked_open_function = mock.mock_open(read_data=my_tree)
        with mock.patch("builtins.open", mocked_open_function):
            model = KrakenNet(kmer_length, channels, my_tree)

        # weird, but I have to
        model.init_from_database(database, requires_grad=True)

        y = model(X)
        # self.assertTupleEqual((N, model.n_nodes), y.shape)
        obs_classes = torch.argmax(y, 1)
        exp_classes = [1, 6, 0, 6, 6, 4, 6, 8, 7, 2, 3]
        npt.assert_array_equal(obs_classes, exp_classes)


class TestLCANets(unittest.TestCase):

    def _test_simple(self, class_):
        N = 20
        nodes_dmp = '../small_demo/taxonomy/nodes.dmp'
        tree = _read_nodes_dmp(nodes_dmp)

        # nodes includes 0, which is unclassified
        leaves, nodes = _get_leaves_nodes(tree)

        model = class_(tree, leaves, nodes)
        X = torch.randn(N, len(leaves), requires_grad=True)
        y = model(X)
        self.assertTupleEqual((N, len(nodes)), y.shape)
        y.sum().backward()

    def _test_net_perfect(self, class_):
        tree = {
            1: [2, 3],
            2: [4, 5, 6],
            3: [7, 8],
        }
        # nodes includes 0, which is unclassified
        leaves, nodes = _get_leaves_nodes(tree)

        # override leaves to get the desired order
        leaves = [0, 4, 5, 6, 7, 8]
        model = class_(tree, leaves, nodes)
        X = torch.tensor([
            [0, 12, 12, 9, 10, 0],  # LCA is 2
            [0, 11, 11, 9, 10, 0],  # LCA is 2
            [0,  4,  4, 3, 12, 0],  # LCA is 7
            [0,  4,  4, 3,  9, 9],  # LCA is 3
            [9,  0,  0, 0,  0, 0],  # LCA is 0
            [1,  1,  0, 0,  0, 0],  # LCA is 4
            [1,  5,  5, 5,  0, 0],  # LCA is 2
            [1,  0,  0, 9,  0, 9],  # LCA is 1
        ],
            requires_grad=True,
            dtype=torch.float32,
        )
        y = model(X)
        # printing y here can be useful for visualizing what the outputs
        # look like
        # print(y)
        obs_classes = torch.argmax(y, 1)
        exp_classes = [2, 2, 7, 3, 0, 4, 2, 1]
        npt.assert_array_equal(obs_classes, exp_classes)
        self.assertTupleEqual((len(X), len(nodes)), y.shape)
        y.sum().backward()

    def _test_net_inexact(self, class_):
        tree = {
            1: [2, 3],
            2: [4, 5, 6],
            3: [7, 8],
        }
        # nodes includes 0, which is unclassified
        leaves, nodes = _get_leaves_nodes(tree)

        # override leaves to get the desired order
        leaves = [0, 4, 5, 6, 7, 8]
        model = class_(tree, leaves, nodes)
        X = torch.tensor([
            [0, 12, 11.99, 9, 10, 0],  # LCA is 2
            [0, 11, 11, 9, 10, 0],  # LCA is 2
            [0,  4,  4, 3, 12, 0],  # LCA is 7
            [0,  4,  4, 3,  9, 8.8],  # LCA is 3
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

    def test_WeightedLCANet_simple(self):
        self._test_simple(WeightedLCANet)

    def test_WeightedLCANet(self):
        self._test_net_perfect(WeightedLCANet)

    def test_WeightedLCANet_inexact(self):
        self._test_net_inexact(WeightedLCANet)

    def test_MatrixLCANet_simple(self):
        self._test_simple(MatrixLCANet)

    def test_MatrixLCANet(self):
        self._test_net_perfect(MatrixLCANet)

    def test_MatrixLCANet_inexact(self):
        self._test_net_inexact(MatrixLCANet)


class TestRTLClasses(unittest.TestCase):

    def _test_rtl_sums(self, class_):
        c = class_(self.tree, self.leaves, n_nodes=9)
        obs = c(self.counts)
        obs.sum().backward()
        npt.assert_array_almost_equal(obs.detach(), self.expected_rtl_sums)

    def setUp(self):
        self.tree = {
            1: [2, 3],
            2: [4, 5, 6],
            3: [7, 8],
        }
        # nodes includes 0, which is unclassified
        # leaves, nodes = _get_leaves_nodes(tree)

        # override leaves to get the desired order
        self.leaves = [0, 4, 5, 6, 7, 8]
        self.counts = torch.tensor([
        #   [0, 1, 2, 3, 4, 5, 6, 7, 8]  # these are the node numbers
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 0, 0, 1, 2, 0, 1, 0, 1],
            [2, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
            dtype=torch.float32, requires_grad=True,
        )
        self.expected_rtl_sums = torch.tensor([
        #   [0, 4, 5, 6, 7, 8]  # these are the node numbers
            [0, 3, 3, 3, 3, 3],
            [2, 2, 0, 1, 1, 2],
            [2, 0, 0, 0, 0, 0],
        ],
            dtype=torch.float32, requires_grad=False,
        )

    def test_RootToLeafSums(self):
        self._test_rtl_sums(RootToLeafSums)

    def test_MatrixRootToLeafSums(self):
        self._test_rtl_sums(MatrixRootToLeafSums)


if __name__ == '__main__':
    unittest.main()
